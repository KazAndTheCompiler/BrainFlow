"""
Brain Persistence - Save and load brain state across sessions.

Stores synapse weights, neuromodulators, development stage, working memory,
hippocampus memories, and step count so the brain remembers everything.

SAFETY FEATURES:
- Automatic versioned backups before every save
- Keeps the last 10 backups rolling
- If load fails due to neuron mismatch, save is PRESERVED (never overwritten)
- Update-safe: brain_state folder is separate from code - updates never touch it
- Schema validation on load to prevent corruption
"""

import os
import sys
import json
import time
import shutil
import hashlib
import numpy as np
from scipy import sparse
from typing import Dict, Any, List, Optional

# Schema version for backward compatibility
SCHEMA_VERSION = "1.0.0"

# P1: Schema migration table
# Maps (from_version, to_version) -> migration function
# Example: MIGRATIONS[("1.0.0", "1.1.0")] = migrate_1_0_0_to_1_1_0
MIGRATIONS = {}

def register_migration(from_ver: str, to_ver: str):
    """Decorator to register a schema migration function."""
    def decorator(func):
        MIGRATIONS[(from_ver, to_ver)] = func
        return func
    return decorator

@register_migration("1.0.0", "1.1.0")
def migrate_1_0_0_to_1_1_0(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example migration: Add new field with default.
    
    In real usage, this would transform the meta dict
    to match the new schema version.
    """
    meta["new_field"] = "default_value"
    return meta

def migrate_schema(meta: Dict[str, Any], target_version: str = SCHEMA_VERSION) -> Dict[str, Any]:
    """
    P1: Migrate meta dict from its current version to target_version.
    
    Args:
        meta: The meta dict loaded from file
        target_version: The schema version to migrate to
        
    Returns:
        Migrated meta dict
        
    Raises:
        ValueError: If no migration path exists
    """
    current_version = meta.get("schema_version", "1.0.0")
    
    if current_version == target_version:
        return meta
    
    # Simple linear migration path
    # In production, you'd want a proper graph traversal
    migration_key = (current_version, target_version)
    
    if migration_key not in MIGRATIONS:
        raise ValueError(
            f"No migration path from {current_version} to {target_version}. "
            f"Available migrations: {list(MIGRATIONS.keys())}"
        )
    
    migration_func = MIGRATIONS[migration_key]
    migrated = migration_func(meta.copy())
    migrated["schema_version"] = target_version
    
    return migrated

# Required fields in meta.json
REQUIRED_META_FIELDS = {
    "step_count": int,
    "total_neurons": int,
    "development_stage": str,
    "neuromodulators": dict,
    "saved_at": (int, float),
}

# Optional fields with defaults
OPTIONAL_META_FIELDS = {
    "uptime": 0.0,
    "schema_version": SCHEMA_VERSION,
    "checksum": "",
}


def _app_root():
    """Directory where brain_state/ should live.

    When frozen by PyInstaller, this is next to the .exe (so users can back up
    their data without spelunking into _internal/). Otherwise it's the project
    root (parent of the `brain/` package).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SAVE_DIR = os.path.join(_app_root(), "brain_state")
BACKUP_DIR = os.path.join(SAVE_DIR, "backups")
MAX_BACKUPS = 10

# If a load is rejected, we set this flag and the server won't auto-save
# over the preserved state until the user explicitly opts in.
_SAVE_LOCKED = False
_LOCK_REASON = ""


def _compute_checksum(data: Dict[str, Any]) -> str:
    """Compute a checksum for data integrity verification."""
    # Remove checksum field before computing
    data_copy = {k: v for k, v in data.items() if k != "checksum"}
    json_str = json.dumps(data_copy, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def _validate_meta(meta: Dict[str, Any]) -> tuple[bool, str]:
    """Validate meta.json structure and types.
    
    Returns: (is_valid, error_message)
    """
    # Check required fields
    for field, expected_type in REQUIRED_META_FIELDS.items():
        if field not in meta:
            return False, f"Missing required field: {field}"
        if not isinstance(meta[field], expected_type):
            return False, f"Invalid type for {field}: expected {expected_type.__name__}, got {type(meta[field]).__name__}"
    
    # Validate neuromodulators structure
    nm = meta.get("neuromodulators", {})
    required_nm = ["dopamine", "serotonin", "norepinephrine", "acetylcholine"]
    for nm_field in required_nm:
        if nm_field not in nm:
            return False, f"Missing neuromodulator: {nm_field}"
        if not isinstance(nm[nm_field], (int, float)):
            return False, f"Invalid type for {nm_field}: expected number"
    
    # Validate checksum if present
    if "checksum" in meta and meta["checksum"]:
        expected = _compute_checksum(meta)
        if meta["checksum"] != expected:
            return False, "Checksum mismatch - data may be corrupted"
    
    return True, ""


def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and normalize meta data."""
    # Add defaults for optional fields
    for field, default in OPTIONAL_META_FIELDS.items():
        if field not in meta:
            meta[field] = default
    
    # Ensure numeric fields are within reasonable bounds
    meta["step_count"] = max(0, int(meta.get("step_count", 0)))
    meta["total_neurons"] = max(1, int(meta.get("total_neurons", 1)))
    meta["uptime"] = max(0.0, float(meta.get("uptime", 0.0)))
    
    # Clamp neuromodulator values to [0, 1]
    if "neuromodulators" in meta:
        for nm in ["dopamine", "serotonin", "norepinephrine", "acetylcholine"]:
            if nm in meta["neuromodulators"]:
                meta["neuromodulators"][nm] = max(0.0, min(1.0, float(meta["neuromodulators"][nm])))
    
    return meta


def is_save_locked():
    """Check if auto-save is locked (protects preserved state)."""
    return _SAVE_LOCKED


def get_lock_reason():
    return _LOCK_REASON


def unlock_save(user_consent=False):
    """Manually unlock saving. Call this after user confirms they want to overwrite."""
    global _SAVE_LOCKED, _LOCK_REASON
    if user_consent:
        _SAVE_LOCKED = False
        _LOCK_REASON = ""
        print("[PERSIST] Save lock released - fresh brain will be saved")
        return True
    return False


def ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def _backup_current_state():
    """Create a timestamped backup of the current brain_state before overwriting."""
    meta_path = os.path.join(SAVE_DIR, "meta.json")
    if not os.path.exists(meta_path):
        return None  # Nothing to back up

    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)

        # Read existing meta to include stage/step in backup name
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            stage = meta.get("development_stage", "unknown")
            steps = meta.get("step_count", 0)
            neurons = meta.get("total_neurons", 0)
        except Exception:
            stage = "unknown"
            steps = 0
            neurons = 0

        # Timestamp folder name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{stage}_{steps}steps_{neurons}n"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        os.makedirs(backup_path, exist_ok=True)

        # Copy meta, regions, synapses (skip backups dir itself)
        for item in os.listdir(SAVE_DIR):
            if item == "backups":
                continue
            src = os.path.join(SAVE_DIR, item)
            dst = os.path.join(backup_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        print(f"[PERSIST] Backup created: {backup_name}")

        # Rotate - keep only the last MAX_BACKUPS
        _rotate_backups()

        return backup_path
    except Exception as e:
        print(f"[PERSIST] Backup failed (continuing anyway): {e}")
        return None


def _rotate_backups():
    """Keep only the last MAX_BACKUPS backups."""
    try:
        if not os.path.exists(BACKUP_DIR):
            return
        backups = sorted([
            os.path.join(BACKUP_DIR, d) for d in os.listdir(BACKUP_DIR)
            if os.path.isdir(os.path.join(BACKUP_DIR, d))
        ])
        while len(backups) > MAX_BACKUPS:
            oldest = backups.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
    except Exception:
        pass


def list_backups():
    """List all available backups."""
    if not os.path.exists(BACKUP_DIR):
        return []
    backups = []
    for d in sorted(os.listdir(BACKUP_DIR), reverse=True):
        full = os.path.join(BACKUP_DIR, d)
        if os.path.isdir(full):
            meta_file = os.path.join(full, "meta.json")
            info = {"name": d, "path": full}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r") as f:
                        info.update(json.load(f))
                except Exception:
                    pass
            backups.append(info)
    return backups


def restore_backup(backup_name):
    """Restore a specific backup as the current brain_state."""
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.isdir(backup_path):
        print(f"[PERSIST] Backup not found: {backup_name}")
        return False

    # First, back up whatever's currently in SAVE_DIR
    _backup_current_state()

    # Clear current state (except backups folder)
    for item in os.listdir(SAVE_DIR):
        if item == "backups":
            continue
        src = os.path.join(SAVE_DIR, item)
        if os.path.isdir(src):
            shutil.rmtree(src, ignore_errors=True)
        else:
            os.remove(src)

    # Copy backup content into SAVE_DIR
    for item in os.listdir(backup_path):
        src = os.path.join(backup_path, item)
        dst = os.path.join(SAVE_DIR, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"[PERSIST] Restored backup: {backup_name}")
    return True


def save_brain(brain, path=None, force=False):
    """Save complete brain state to disk.

    Args:
        brain: Brain instance to save
        path: Save directory (default: brain_state/)
        force: If True, save even when save is locked (user-initiated save)
    """
    if path is None:
        path = SAVE_DIR
    ensure_save_dir()

    # Check lock - prevents auto-save from overwriting preserved state
    if _SAVE_LOCKED and not force:
        print(f"[PERSIST] Save skipped - state is locked: {_LOCK_REASON}")
        print(f"[PERSIST] Use /api/brain/unlock or restart with matching neuron count to enable saves")
        return False

    # Back up existing state before overwriting (rolling versioned backups)
    _backup_current_state()

    state = {
        "step_count": brain.step_count,
        "total_neurons": brain.total_neurons,
        "development_stage": brain.development_stage,
        "neuromodulators": brain.neuromodulators,
        "saved_at": time.time(),
        "uptime": time.time() - brain.start_time,
        "schema_version": SCHEMA_VERSION,
    }
    
    # Add checksum for integrity verification
    state["checksum"] = _compute_checksum(state)

    # Save metadata
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(state, f, indent=2)

    # Save synapse weights (sparse matrices)
    syn_dir = os.path.join(path, "synapses")
    os.makedirs(syn_dir, exist_ok=True)
    for (src, dst), syn in brain.connections.items():
        if syn.nnz > 0:
            fname = f"{src}__{dst}.npz"
            sparse.save_npz(os.path.join(syn_dir, fname), syn.weights)
            # Save traces
            np.savez_compressed(
                os.path.join(syn_dir, f"{src}__{dst}_traces.npz"),
                pre_trace=syn.pre_trace,
                post_trace=syn.post_trace,
                modulation=np.array([syn.modulation]),
            )

    # Save region-specific state
    reg_dir = os.path.join(path, "regions")
    os.makedirs(reg_dir, exist_ok=True)

    for name, region in brain.regions.items():
        reg_state = {}

        # Save neuron state
        reg_state["v"] = region.neurons.v.tolist()
        reg_state["u"] = region.neurons.u.tolist()

        # Region-specific data
        if name == "hippocampus" and hasattr(region, "memory_buffer"):
            memories = [m.tolist() for m in region.memory_buffer[-100:]]  # Last 100
            reg_state["memory_buffer"] = memories

        if name == "prefrontal" and hasattr(region, "working_memory"):
            reg_state["working_memory"] = region.working_memory.tolist()

        if name == "predictive" and hasattr(region, "prediction"):
            reg_state["prediction"] = region.prediction.tolist()

        if name == "association" and hasattr(region, "binding_strength"):
            reg_state["binding_strength"] = region.binding_strength.tolist()

        if name == "brainstem":
            reg_state["energy"] = float(region.energy)
            reg_state["arousal"] = float(region.arousal)

        with open(os.path.join(reg_dir, f"{name}.json"), "w") as f:
            json.dump(reg_state, f)

    print(f"[PERSIST] Brain saved ({brain.step_count} steps, {len(brain.connections)} connections)")
    return True


def load_brain(brain, path=None):
    """Load brain state from disk. Returns True if successful."""
    global _SAVE_LOCKED, _LOCK_REASON

    if path is None:
        path = SAVE_DIR

    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        print("[PERSIST] No saved state found")
        return False

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Validate schema
        is_valid, error_msg = _validate_meta(meta)
        if not is_valid:
            print(f"[PERSIST] Schema validation failed: {error_msg}")
            print("[PERSIST] *** SAVE LOCKED *** Corrupted state will NOT be overwritten.")
            _SAVE_LOCKED = True
            _LOCK_REASON = f"Schema validation failed: {error_msg}"
            return False
        
        # Sanitize/normalize the data
        meta = _sanitize_meta(meta)
        
        # P1: Migrate schema if needed
        try:
            if meta.get("schema_version", "1.0.0") != SCHEMA_VERSION:
                print(f"[PERSIST] Migrating schema from {meta.get('schema_version', '1.0.0')} to {SCHEMA_VERSION}")
                meta = migrate_schema(meta, SCHEMA_VERSION)
                print(f"[PERSIST] Schema migration complete")
        except ValueError as e:
            print(f"[PERSIST] Schema migration failed: {e}")
            print(f"[PERSIST] *** SAVE LOCKED *** Cannot load state with incompatible schema.")
            _SAVE_LOCKED = True
            _LOCK_REASON = f"Schema migration failed: {e}"
            return False

        # Verify compatibility
        if meta["total_neurons"] != brain.total_neurons:
            print(f"[PERSIST] Neuron count mismatch: saved={meta['total_neurons']}, current={brain.total_neurons}")
            print(f"[PERSIST] *** SAVE LOCKED *** Preserved state will NOT be overwritten.")
            print(f"[PERSIST] To restore this state, restart with: python run.py --neurons {meta['total_neurons']}")
            print(f"[PERSIST] Stage was: {meta.get('development_stage', '?')}, Steps: {meta.get('step_count', 0):,}")

            # Create a safety backup of the preserved state
            _backup_current_state()

            # Lock saves so this precious state isn't overwritten
            _SAVE_LOCKED = True
            _LOCK_REASON = (
                f"Saved state has {meta['total_neurons']:,} neurons but brain is running "
                f"with {brain.total_neurons:,}. Restart with --neurons {meta['total_neurons']} "
                f"to restore, or call /api/brain/unlock to discard."
            )
            return False

        # Restore metadata
        brain.step_count = meta["step_count"]
        brain.development_stage = meta["development_stage"]
        brain.neuromodulators = meta["neuromodulators"]

        # Restore synapse weights
        syn_dir = os.path.join(path, "synapses")
        if os.path.exists(syn_dir):
            for (src, dst), syn in brain.connections.items():
                fname = os.path.join(syn_dir, f"{src}__{dst}.npz")
                if os.path.exists(fname) and syn.nnz > 0:
                    loaded = sparse.load_npz(fname)
                    if loaded.shape == syn.weights.shape:
                        syn.weights = loaded
                    traces_path = os.path.join(syn_dir, f"{src}__{dst}_traces.npz")
                    if os.path.exists(traces_path):
                        traces = np.load(traces_path)
                        syn.pre_trace = traces["pre_trace"]
                        syn.post_trace = traces["post_trace"]
                        syn.modulation = float(traces["modulation"][0])

        # Restore region state
        reg_dir = os.path.join(path, "regions")
        if os.path.exists(reg_dir):
            for name, region in brain.regions.items():
                reg_path = os.path.join(reg_dir, f"{name}.json")
                if not os.path.exists(reg_path):
                    continue
                with open(reg_path, "r") as f:
                    reg_state = json.load(f)

                # Restore neuron state
                if "v" in reg_state:
                    v = np.array(reg_state["v"], dtype=np.float64)
                    if len(v) == region.n_neurons:
                        region.neurons.v = v
                if "u" in reg_state:
                    u = np.array(reg_state["u"], dtype=np.float64)
                    if len(u) == region.n_neurons:
                        region.neurons.u = u

                # Region-specific
                if name == "hippocampus" and "memory_buffer" in reg_state:
                    region.memory_buffer = [
                        np.array(m, dtype=bool) for m in reg_state["memory_buffer"]
                    ]

                if name == "prefrontal" and "working_memory" in reg_state:
                    wm = np.array(reg_state["working_memory"], dtype=np.float32)
                    if len(wm) == region.n_neurons:
                        region.working_memory = wm

                if name == "predictive" and "prediction" in reg_state:
                    pred = np.array(reg_state["prediction"], dtype=np.float32)
                    if len(pred) == region.n_neurons:
                        region.prediction = pred

                if name == "association" and "binding_strength" in reg_state:
                    bs = np.array(reg_state["binding_strength"], dtype=np.float64)
                    if len(bs) == region.n_neurons:
                        region.binding_strength = bs

                if name == "brainstem":
                    if "energy" in reg_state:
                        region.energy = reg_state["energy"]
                    if "arousal" in reg_state:
                        region.arousal = reg_state["arousal"]

        print(f"[PERSIST] Brain loaded (step {brain.step_count}, stage {brain.development_stage})")
        return True

    except Exception as e:
        print(f"[PERSIST] Load error: {e}")
        return False


def get_save_info(path=None):
    """Get info about saved brain state without loading it."""
    if path is None:
        path = SAVE_DIR
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)
