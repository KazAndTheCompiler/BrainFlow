"""
NeuroLinked Brain Configuration
All hyperparameters for the neuromorphic brain system.
"""

import os
import secrets
import numpy as np


class BrainConfig:
    # --- Determinism / RNG Seeding ---
    # Set NEUROLINKED_SEED for reproducible runs (testing, debugging)
    # Leave unset for random initialization
    RNG_SEED = os.environ.get("NEUROLINKED_SEED", None)
    
    @classmethod
    def get_rng(cls):
        """Get a seeded numpy random Generator for reproducibility."""
        if cls.RNG_SEED is not None:
            try:
                seed = int(cls.RNG_SEED)
                return np.random.default_rng(seed)
            except (ValueError, TypeError):
                pass
        return np.random.default_rng()
    # --- Scale ---
    TOTAL_NEURONS = 100_000  # Default 100K, set to 1_000_000 for full scale
    SYNAPSES_PER_NEURON = 1200  # Average connections per neuron

    # --- Region proportions (must sum to 1.0) ---
    REGION_PROPORTIONS = {
        "sensory_cortex":    0.107,
        "feature_layer":     0.101,
        "association":       0.372,
        "concept_layer":     0.038,
        "predictive":        0.156,
        "motor_cortex":      0.025,
        "cerebellum":        0.075,
        "reflex_arc":        0.046,
        "brainstem":         0.027,
        "hippocampus":       0.033,
        "prefrontal":        0.020,
    }

    # --- Izhikevich neuron parameters by region ---
    # (a, b, c, d) - different firing patterns per region
    NEURON_PARAMS = {
        "sensory_cortex":  {"a": 0.02, "b": 0.2, "c": -65, "d": 8},     # Regular spiking
        "feature_layer":   {"a": 0.02, "b": 0.25, "c": -65, "d": 8},    # Regular spiking
        "association":     {"a": 0.02, "b": 0.2, "c": -50, "d": 2},     # Chattering
        "concept_layer":   {"a": 0.02, "b": 0.2, "c": -55, "d": 4},    # Intrinsically bursting
        "predictive":      {"a": 0.1, "b": 0.2, "c": -65, "d": 2},     # Fast spiking
        "motor_cortex":    {"a": 0.02, "b": 0.2, "c": -65, "d": 8},    # Regular spiking
        "cerebellum":      {"a": 0.1, "b": 0.2, "c": -65, "d": 2},     # Fast spiking
        "reflex_arc":      {"a": 0.1, "b": 0.26, "c": -65, "d": 2},    # Fast spiking
        "brainstem":       {"a": 0.02, "b": 0.25, "c": -65, "d": 0.05}, # Low-threshold
        "hippocampus":     {"a": 0.02, "b": 0.2, "c": -50, "d": 2},    # Chattering
        "prefrontal":      {"a": 0.02, "b": 0.2, "c": -55, "d": 4},    # Intrinsically bursting
    }

    # --- Connectivity matrix (source -> target probability) ---
    # Sparse: only define non-zero connections
    CONNECTIVITY = {
        ("sensory_cortex", "feature_layer"):  0.15,
        ("sensory_cortex", "reflex_arc"):     0.10,
        ("feature_layer", "association"):     0.20,
        ("feature_layer", "concept_layer"):   0.08,
        ("association", "concept_layer"):     0.12,
        ("association", "predictive"):        0.15,
        ("association", "hippocampus"):       0.10,
        ("association", "prefrontal"):        0.08,
        ("concept_layer", "association"):     0.10,
        ("concept_layer", "predictive"):      0.10,
        ("concept_layer", "prefrontal"):      0.08,
        ("predictive", "association"):        0.12,
        ("predictive", "sensory_cortex"):     0.05,
        ("predictive", "motor_cortex"):       0.06,
        ("prefrontal", "motor_cortex"):       0.15,
        ("prefrontal", "association"):        0.08,
        ("prefrontal", "predictive"):         0.06,
        ("motor_cortex", "cerebellum"):       0.12,
        ("motor_cortex", "brainstem"):        0.10,
        ("cerebellum", "motor_cortex"):       0.10,
        ("cerebellum", "brainstem"):          0.05,
        ("reflex_arc", "motor_cortex"):       0.15,
        ("reflex_arc", "brainstem"):          0.08,
        ("brainstem", "sensory_cortex"):      0.03,
        ("brainstem", "motor_cortex"):        0.05,
        ("hippocampus", "association"):       0.12,
        ("hippocampus", "prefrontal"):        0.08,
        ("hippocampus", "concept_layer"):     0.06,
    }

    # --- STDP Learning ---
    STDP_TAU_PLUS = 20.0     # ms, LTP time constant
    STDP_TAU_MINUS = 20.0    # ms, LTD time constant
    STDP_A_PLUS = 0.01       # LTP amplitude
    STDP_A_MINUS = 0.012     # LTD amplitude (slightly stronger for stability)
    STDP_W_MAX = 1.0         # Maximum synapse weight
    STDP_W_MIN = 0.0         # Minimum synapse weight

    # --- Neuromodulators ---
    DOPAMINE_BASELINE = 0.5
    ACETYLCHOLINE_BASELINE = 0.5
    NOREPINEPHRINE_BASELINE = 0.3
    SEROTONIN_BASELINE = 0.5

    # --- Simulation ---
    DT = 1.0                 # Timestep in ms
    STEPS_PER_UPDATE = 10    # Steps per WebSocket update
    THALAMIC_NOISE = 5.0     # Background noise amplitude

    # --- Safety kernel ---
    SAFETY_FORCE_LIMIT = 100.0
    SAFETY_VELOCITY_LIMIT = 50.0
    SAFETY_COLLISION_MARGIN = 0.1

    # --- Development stages ---
    STAGES = {
        "EMBRYONIC":    (0, 100_000),
        "JUVENILE":     (100_000, 2_000_000),
        "ADOLESCENT":   (2_000_000, 10_000_000),
        "MATURE":       (10_000_000, float("inf")),
    }

    # --- Server ---
    HOST = "0.0.0.0"
    PORT = 8000
    WS_UPDATE_RATE = 30  # Hz

    # --- Security (Production-Grade) ---
    # SECURITY WARNING: These defaults are for development only.
    # In production, ALWAYS set NEUROLINKED_API_TOKEN via environment variable.
    
    # Master API token for MCP server and HTTP API authentication
    # Generate a secure token with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    API_TOKEN = os.environ.get("NEUROLINKED_API_TOKEN", "")
    
    # Token file path (alternative to env var)
    TOKEN_FILE = os.environ.get("NEUROLINKED_TOKEN_FILE", "")
    
    # Require authentication for all API endpoints (except health check)
    REQUIRE_AUTH = os.environ.get("NEUROLINKED_REQUIRE_AUTH", "true").lower() == "true"
    
    # CORS origins (production: restrict to your domain)
    CORS_ORIGINS = os.environ.get("NEUROLINKED_CORS_ORIGINS", "http://localhost:8000").split(",")
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.environ.get("NEUROLINKED_RATE_LIMIT", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.environ.get("NEUROLINKED_RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.environ.get("NEUROLINKED_RATE_LIMIT_WINDOW", "60"))
    
    # Bind address (production: bind to localhost only or use unix socket)
    HOST = os.environ.get("NEUROLINKED_HOST", "127.0.0.1")
    PORT = int(os.environ.get("NEUROLINKED_PORT", "8000"))
    
    @classmethod
    def get_api_token(cls) -> str:
        """Get API token from env var or token file."""
        if cls.API_TOKEN:
            return cls.API_TOKEN
        if cls.TOKEN_FILE and os.path.exists(cls.TOKEN_FILE):
            try:
                with open(cls.TOKEN_FILE, 'r') as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""
    
    @classmethod
    def generate_secure_token(cls) -> str:
        """Generate a new secure API token."""
        return secrets.token_urlsafe(32)
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return os.environ.get("NEUROLINKED_ENV", "development").lower() == "production"
    
    @classmethod
    def validate_token(cls, token: str) -> bool:
        """Validate an API token against the configured token."""
        expected = cls.get_api_token()
        if not expected:
            # No token configured - reject all requests in production
            return not cls.is_production()
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(token, expected)
    
    # --- P1: Knowledge Store Retention ---
    # Maximum entries in knowledge store (oldest auto-pruned)
    KNOWLEDGE_MAX_ENTRIES = int(os.environ.get("NEUROLINKED_KNOWLEDGE_MAX_ENTRIES", "10000"))
    
    # Knowledge retention days (entries older than this are pruned)
    KNOWLEDGE_RETENTION_DAYS = int(os.environ.get("NEUROLINKED_KNOWLEDGE_RETENTION_DAYS", "90"))
    
    # --- P1: Screen Observer Privacy ---
    # Window titles to exclude from screen capture (privacy filter)
    SCREEN_EXCLUDED_TITLES = os.environ.get(
        "NEUROLINKED_SCREEN_EXCLUDED_TITLES",
        "password,pass,secret,login,credential,bitwarden,1password,lastpass,dashlane"
    ).lower().split(",")
    
    # Text patterns to redact from OCR (PII protection)
    SCREEN_REDACT_PATTERNS = [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit cards
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'password[:\s]+\S+',  # Password fields
        r'secret[:\s]+\S+',  # Secret fields
        r'token[:\s]+\S+',  # Token fields
        r'key[:\s]+\S+',  # Key fields
    ]
    
    # --- P1: Rate Limiting Backend ---
    # Use Redis for distributed rate limiting (empty = in-memory)
    REDIS_URL = os.environ.get("NEUROLINKED_REDIS_URL", "")
    
    # --- P1: Activity Log Retention ---
    # Maximum activity log entries (ClaudeBridge)
    ACTIVITY_LOG_MAX_ENTRIES = int(os.environ.get("NEUROLINKED_ACTIVITY_LOG_MAX", "1000"))
    
    # --- P1: Metrics ---
    # Enable Prometheus metrics endpoint
    METRICS_ENABLED = os.environ.get("NEUROLINKED_METRICS_ENABLED", "false").lower() == "true"
    METRICS_PORT = int(os.environ.get("NEUROLINKED_METRICS_PORT", "9090"))
