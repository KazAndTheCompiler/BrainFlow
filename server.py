"""
FastAPI + WebSocket Server for NeuroLinked Brain

Serves the 3D dashboard and streams real-time brain state via WebSocket.
Provides Claude integration API for reading brain state and sending input.
"""

import asyncio
import json
import os
import sys
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from brain.brain import Brain
from brain.config import BrainConfig
from brain.persistence import (
    save_brain, load_brain, get_save_info,
    list_backups, restore_backup, is_save_locked, get_lock_reason, unlock_save,
)
from brain.claude_bridge import ClaudeBridge
from brain.screen_observer import ScreenObserver
from brain.video_recorder import VideoRecorder
from sensory.text import TextEncoder
from sensory.vision import VisionEncoder
from sensory.audio import AudioEncoder
from sensory.screen_ui import ScreenUIEngine

app = FastAPI(title="NeuroLinked Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
brain: Brain = None
text_encoder: TextEncoder = None
vision_encoder: VisionEncoder = None
audio_encoder: AudioEncoder = None
claude_bridge: ClaudeBridge = None
screen_observer: ScreenObserver = None
screen_ui_engine: ScreenUIEngine = None
video_recorder: VideoRecorder = None

# Simulation thread control
sim_running = False
sim_thread = None
connected_clients = set()

# Auto-save interval (every 5 minutes)
AUTO_SAVE_INTERVAL = 300
_last_auto_save = 0


def init_brain():
    """Initialize the brain and sensory encoders."""
    global brain, text_encoder, vision_encoder, audio_encoder, claude_bridge, screen_observer, screen_ui_engine, video_recorder
    brain = Brain()
    text_encoder = TextEncoder(feature_dim=256)
    vision_encoder = VisionEncoder(feature_dim=256)
    audio_encoder = AudioEncoder(feature_dim=256)
    claude_bridge = ClaudeBridge(brain)
    screen_observer = ScreenObserver(feature_dim=256, capture_interval=2.0)
    screen_ui_engine = ScreenUIEngine()
    screen_observer.attach_brain(
        brain=brain,
        text_encoder=text_encoder,
        knowledge_store=claude_bridge.knowledge,
        screen_ui_engine=screen_ui_engine,
    )
    # Video recorder saves screen to .mp4 segments (off by default)
    video_recorder = VideoRecorder(fps=10, segment_minutes=10)

    # Try to load saved state
    loaded = load_brain(brain)
    if loaded:
        print("[SERVER] Restored brain from saved state")
    else:
        print("[SERVER] Starting fresh brain")


_last_screen_log = 0
_BRAIN_EVENT_LOG_INTERVAL = 15
SCREEN_LOG_INTERVAL = 30

def simulation_loop():
    """Run brain simulation in background thread."""
    global sim_running, _last_auto_save, _last_screen_log, _last_brain_event_log
    target_dt = 1.0 / 100  # Target 100 steps/sec
    while sim_running:
        start = time.time()
        try:
            now = time.time()

            if now - _last_screen_log > SCREEN_LOG_INTERVAL and claude_bridge:
                try:
                    state = screen_observer.get_state()
                    if state.get("ui_pipeline_active") and state.get("last_ui_summary"):
                        claude_bridge.knowledge.store(
                            text=f"Screen: {state['last_ui_summary']} | "
                                 f"Regions: {state.get('ui_regions', 0)} | "
                                 f"Step: {brain.step_count}",
                            source="brain_monitor",
                            tags=["brain", "monitor", "screen"],
                        )
                    else:
                        claude_bridge.knowledge.store(
                            text=f"Brain step {brain.step_count} | "
                                 f"Stage: {brain.development_stage} | "
                                 f"Rate: {brain.steps_per_second:.0f} Hz",
                            source="brain_monitor",
                            tags=["brain", "monitor", "heartbeat"],
                        )
                except Exception:
                    pass
                _last_screen_log = now

            if now - _last_brain_event_log > _BRAIN_EVENT_LOG_INTERVAL and claude_bridge:
                try:
                    events = claude_bridge.poll_brain_events()
                    for event in events:
                        claude_bridge.knowledge.store(
                            text=f"[EVENT] {event['message']}",
                            source="brain_event",
                            tags=["brain", "event", event["type"]],
                            metadata={"event_type": event["type"], "step": event["step"]},
                        )
                except Exception:
                    pass
                _last_brain_event_log = now

            brain.step()

            # Auto-save periodically
            now = time.time()
            if now - _last_auto_save > AUTO_SAVE_INTERVAL:
                try:
                    save_brain(brain)
                    _last_auto_save = now
                except Exception as e:
                    print(f"[SERVER] Auto-save error: {e}")

        except Exception as e:
            print(f"[SIM] Error: {e}")
        elapsed = time.time() - start
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_simulation():
    """Start the background simulation thread."""
    global sim_running, sim_thread
    if sim_running:
        return
    sim_running = True
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    print("[SERVER] Simulation started")


def stop_simulation():
    """Stop the background simulation thread."""
    global sim_running
    sim_running = False
    print("[SERVER] Simulation stopped")


# --- Static files ---
# When frozen by PyInstaller, dashboard lives next to the .exe, not next to this file.
if getattr(sys, "frozen", False):
    _base_dir = os.path.dirname(sys.executable)
else:
    _base_dir = os.path.dirname(__file__)
dashboard_path = os.path.join(_base_dir, "dashboard")
# Fallback: if the user-editable dashboard folder is missing, look inside the bundle.
if not os.path.isdir(dashboard_path):
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
app.mount("/css", StaticFiles(directory=os.path.join(dashboard_path, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(dashboard_path, "js")), name="js")


@app.on_event("startup")
async def startup():
    init_brain()
    start_simulation()


@app.on_event("shutdown")
async def shutdown():
    # Save brain state on shutdown
    try:
        save_brain(brain)
        print("[SERVER] Brain saved on shutdown")
    except Exception as e:
        print(f"[SERVER] Save on shutdown failed: {e}")

    stop_simulation()
    if screen_observer:
        screen_observer.stop()
    if video_recorder:
        video_recorder.stop()
    if vision_encoder:
        vision_encoder.stop_webcam()
    if audio_encoder:
        audio_encoder.stop_microphone()


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse(os.path.join(dashboard_path, "index.html"))


@app.get("/api/state")
async def get_state():
    return JSONResponse(brain.get_state())


@app.get("/api/positions")
async def get_positions():
    return JSONResponse(brain.get_neuron_positions())


@app.post("/api/input/text")
async def input_text(data: dict):
    text = data.get("text", "")
    if text:
        features = text_encoder.encode(text)
        brain.inject_sensory_input("text", features)
        # Also log to Claude bridge
        if claude_bridge:
            claude_bridge.send_observation({
                "type": "text",
                "content": text,
                "source": "user",
            })
    return {"status": "ok", "encoded_dim": len(features) if text else 0}


@app.post("/api/input/vision/start")
async def start_vision():
    success = vision_encoder.start_webcam()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/input/vision/stop")
async def stop_vision():
    vision_encoder.stop_webcam()
    return {"status": "stopped"}


@app.post("/api/input/audio/start")
async def start_audio():
    success = audio_encoder.start_microphone()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/input/audio/stop")
async def stop_audio():
    audio_encoder.stop_microphone()
    return {"status": "stopped"}


@app.post("/api/control/pause")
async def pause():
    stop_simulation()
    return {"status": "paused"}


@app.post("/api/control/resume")
async def resume():
    start_simulation()
    return {"status": "running"}


@app.post("/api/control/reset")
async def reset():
    stop_simulation()
    init_brain()
    start_simulation()
    return {"status": "reset"}


# =============================================================================
# Claude Integration API
# =============================================================================

@app.get("/api/claude/summary")
async def claude_summary():
    """Primary endpoint for Claude to read brain state."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_brain_summary()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/insights")
async def claude_insights():
    """Get brain-derived insights useful for Claude."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_insights()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/claude/observe")
async def claude_observe(data: dict):
    """
    Claude sends an observation to the brain.
    Body: {"type": "text"|"action"|"context", "content": "...", "source": "claude"}
    """
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    claude_bridge.send_observation(data)
    return {"status": "ok", "interaction_count": claude_bridge._interaction_count}


@app.get("/api/claude/status")
async def claude_status():
    """Get Claude bridge connection status."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    state = claude_bridge.get_state()
    if screen_observer:
        state["screen_observer"] = screen_observer.get_state()
    if video_recorder:
        state["video_recorder"] = video_recorder.get_state()
    return JSONResponse(state)


@app.get("/api/claude/activity")
async def claude_activity():
    """Get recent activity log."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    return JSONResponse(claude_bridge.get_activity_log())


@app.get("/api/claude/learned")
async def claude_learned():
    """Get what the brain has learned - grouped patterns and associations."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_learned_patterns()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/learned/summary")
async def claude_learned_summary():
    """Get plain-English summary of what the brain has learned."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        text = claude_bridge.get_learning_summary()
        return JSONResponse({"summary": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Knowledge Store API (text storage & retrieval — replaces Obsidian)
# =============================================================================

@app.get("/api/claude/recall")
async def claude_recall(q: str = "", limit: int = 10):
    """Recall knowledge about a specific topic."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.recall(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/search")
async def claude_search(q: str = "", limit: int = 20):
    """Full-text search across all stored knowledge."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.search_knowledge(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/semantic")
async def claude_semantic(q: str = "", limit: int = 10):
    """Semantic (associative) search - finds conceptually related memories
    via TF-IDF cosine similarity, not just keyword matching."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.knowledge.semantic_search(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results),
                             "mode": "semantic_tfidf"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/knowledge")
async def claude_knowledge():
    """Get knowledge store stats and recent entries."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        stats = claude_bridge.get_knowledge_stats()
        recent = claude_bridge.get_recent_knowledge(limit=10)
        return JSONResponse({"stats": stats, "recent": recent})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/claude/remember")
async def claude_remember(data: dict):
    """
    Store a piece of knowledge directly.
    Body: {"text": "...", "source": "claude", "tags": ["optional", "tags"]}
    """
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "text field is required"}, status_code=400)
    source = data.get("source", "claude")
    tags = data.get("tags", None)
    try:
        entry_id = claude_bridge.store_knowledge(text=text, source=source, tags=tags)
        return {"status": "stored", "id": entry_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Screen Observation API
# =============================================================================

@app.post("/api/screen/start")
async def start_screen():
    """Start screen observation."""
    if not screen_observer:
        return JSONResponse({"error": "Screen observer not initialized"}, status_code=503)
    success = screen_observer.start()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/screen/stop")
async def stop_screen():
    """Stop screen observation."""
    if screen_observer:
        screen_observer.stop()
    return {"status": "stopped"}


@app.get("/api/screen/state")
async def screen_state():
    """Get screen observer state."""
    if not screen_observer:
        return JSONResponse({"error": "Screen observer not initialized"}, status_code=503)
    return JSONResponse(screen_observer.get_state())


# =============================================================================
# Screen UI Analysis API
# =============================================================================

@app.post("/api/screen/analyze")
async def analyze_screen():
    """Run full UI analysis on current screen (one-shot, does not start observation)."""
    if not screen_ui_engine:
        return JSONResponse({"error": "ScreenUIEngine not initialized"}, status_code=503)
    if not screen_observer:
        return JSONResponse({"error": "Screen observer not initialized"}, status_code=503)

    try:
        screenshot = screen_observer._capture_screen()
        if screenshot is None:
            return JSONResponse({"error": "Screen capture failed"}, status_code=500)

        window_title = screen_observer._get_window_title()
        layout = screen_ui_engine.analyze(
            screenshot,
            capture_num=screen_observer._capture_count,
            window_title=window_title
        )

        summary = screen_ui_engine.get_text_summary(layout)

        regions = []
        for r in layout.regions:
            regions.append({
                "type": r.region_type.value,
                "bounds": r.bounds,
                "text": r.text[:200] if r.text else "",
                "label": r.label,
                "interactive": r.is_interactive,
                "confidence": round(r.confidence, 2),
            })

        salient = layout.saliency_region
        salient_info = None
        if salient:
            salient_info = {
                "type": salient.region_type.value,
                "bounds": salient.bounds,
                "text": salient.text[:200] if salient.text else "",
                "label": salient.label,
            }

        return JSONResponse({
            "capture_num": layout.capture_num,
            "timestamp": layout.timestamp,
            "window_title": layout.window_title,
            "dominant_type": layout.dominant_type.value,
            "text_chars": len(layout.full_text),
            "region_count": len(layout.regions),
            "summary": summary,
            "salient_region": salient_info,
            "regions": regions[:20],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/screen/inject")
async def inject_screen_layout():
    """Capture screen, analyze UI, and inject structured data into brain."""
    if not screen_ui_engine or not screen_observer or not brain:
        return JSONResponse({"error": "Engines not initialized"}, status_code=503)

    try:
        screenshot = screen_observer._capture_screen()
        if screenshot is None:
            return JSONResponse({"error": "Screen capture failed"}, status_code=500)

        window_title = screen_observer._get_window_title()
        layout = screen_ui_engine.analyze(screenshot, capture_num=screen_observer._capture_count, window_title=window_title)
        encoded = screen_ui_engine.encode_for_brain(layout)

        if encoded["text_features"].size > 0:
            brain.inject_sensory_input("text", encoded["text_features"])
        if encoded["vision_features"].size > 0:
            brain.inject_sensory_input("vision", encoded["vision_features"])

        brain.neuromodulators["acetylcholine"] = min(1.0, brain.neuromodulators.get("acetylcholine", 0.5) + 0.08)
        brain.neuromodulators["norepinephrine"] = min(1.0, brain.neuromodulators.get("norepinephrine", 0.3) + 0.05)

        return JSONResponse({
            "status": "injected",
            "summary": encoded["summary"],
            "text_chars": len(layout.full_text),
            "region_count": len(layout.regions),
            "dominant_type": layout.dominant_type.value,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Video Recording API
# =============================================================================

@app.post("/api/video/start")
async def start_video():
    """Start video recording (saves screen to .mp4 segments)."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    success = video_recorder.start()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/video/stop")
async def stop_video():
    """Stop video recording and close current segment."""
    if video_recorder:
        video_recorder.stop()
    return {"status": "stopped"}


@app.get("/api/video/state")
async def video_state():
    """Get video recorder state (active, fps, disk usage, file count)."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    return JSONResponse(video_recorder.get_state())


@app.get("/api/video/list")
async def video_list():
    """List all recorded .mp4 files with size and timestamps."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    return JSONResponse({"recordings": video_recorder.list_recordings()})


@app.post("/api/video/delete")
async def video_delete(data: dict):
    """Delete a recording by filename. Body: {'name': 'screen_YYYYMMDD_HHMMSS.mp4'}"""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    name = data.get("name")
    if not name:
        return JSONResponse({"error": "Missing 'name' field"}, status_code=400)
    success = video_recorder.delete_recording(name)
    return {"status": "deleted" if success else "not_found", "name": name}


@app.get("/api/video/recording/{filename}")
async def video_download(filename: str):
    """Stream/download a specific recording file."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    # Only allow files in the recordings directory, and only .mp4
    if not filename.endswith(".mp4") or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    path = os.path.join(video_recorder.output_dir, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=filename)


# =============================================================================
# Persistence API
# =============================================================================

@app.post("/api/brain/save")
async def save_state():
    """Save brain state to disk."""
    try:
        save_brain(brain)
        return {"status": "saved", "step": brain.step_count}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/brain/load")
async def load_state():
    """Load brain state from disk."""
    try:
        stop_simulation()
        success = load_brain(brain)
        start_simulation()
        return {"status": "loaded" if success else "no_save_found", "step": brain.step_count}
    except Exception as e:
        start_simulation()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/brain/save-info")
async def save_info():
    """Get info about saved state without loading."""
    info = get_save_info()
    if info:
        return JSONResponse(info)
    return JSONResponse({"saved": False})


@app.get("/api/brain/backups")
async def brain_backups():
    """List all available brain state backups."""
    return JSONResponse({
        "backups": list_backups(),
        "save_locked": is_save_locked(),
        "lock_reason": get_lock_reason(),
    })


@app.post("/api/brain/restore-backup")
async def brain_restore_backup(data: dict):
    """Restore a specific backup. Body: {'name': 'backup_folder_name'}"""
    name = data.get("name", "")
    if not name:
        return JSONResponse({"error": "name field required"}, status_code=400)
    try:
        stop_simulation()
        success = restore_backup(name)
        if success:
            init_brain()
            start_simulation()
            return {"status": "restored", "backup": name, "step": brain.step_count}
        else:
            start_simulation()
            return JSONResponse({"error": "Backup not found"}, status_code=404)
    except Exception as e:
        start_simulation()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/brain/unlock")
async def brain_unlock(data: dict = None):
    """
    Unlock save protection. Required if neuron count mismatch locked saving.
    Body: {'confirm': true} - user must confirm they want to overwrite preserved state
    """
    data = data or {}
    if not data.get("confirm", False):
        return JSONResponse({
            "error": "Confirmation required",
            "message": "Pass {'confirm': true} to acknowledge you want to overwrite preserved state.",
            "lock_reason": get_lock_reason(),
        }, status_code=400)
    unlock_save(user_consent=True)
    return {"status": "unlocked", "warning": "Next save will overwrite preserved state"}


@app.get("/api/brain/lock-status")
async def brain_lock_status():
    """Check if save is currently locked."""
    return JSONResponse({
        "locked": is_save_locked(),
        "reason": get_lock_reason(),
    })


# =============================================================================
# WebSocket for real-time streaming
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"[WS] Client connected ({len(connected_clients)} total)")

    # Send initial neuron positions
    try:
        positions = brain.get_neuron_positions()
        await ws.send_json({"type": "init", "positions": positions})
    except Exception:
        pass

    try:
        update_interval = 1.0 / BrainConfig.WS_UPDATE_RATE
        while True:
            start = time.time()

            # Send brain state
            state = brain.get_state()

            # Add Claude bridge info to state
            if claude_bridge:
                state["claude"] = {
                    "connected": True,
                    "interactions": claude_bridge._interaction_count,
                }
            if screen_observer:
                state["screen_observer"] = screen_observer.get_state()
            if video_recorder:
                state["video_recorder"] = video_recorder.get_state()

            await ws.send_json({"type": "state", "data": state})

            # Check for incoming messages (text input, commands)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                if msg.get("type") == "text_input":
                    features = text_encoder.encode(msg["text"])
                    brain.inject_sensory_input("text", features)
                    if claude_bridge:
                        claude_bridge.send_observation({
                            "type": "text",
                            "content": msg["text"],
                            "source": "dashboard",
                        })
                elif msg.get("type") == "command":
                    cmd = msg.get("cmd")
                    if cmd == "start_vision":
                        vision_encoder.start_webcam()
                    elif cmd == "stop_vision":
                        vision_encoder.stop_webcam()
                    elif cmd == "start_audio":
                        audio_encoder.start_microphone()
                    elif cmd == "stop_audio":
                        audio_encoder.stop_microphone()
                    elif cmd == "start_screen":
                        screen_observer.start()
                    elif cmd == "stop_screen":
                        screen_observer.stop()
                    elif cmd == "start_video":
                        if video_recorder:
                            video_recorder.start()
                    elif cmd == "stop_video":
                        if video_recorder:
                            video_recorder.stop()
                    elif cmd == "save":
                        save_brain(brain)
                    elif cmd == "load":
                        load_brain(brain)
            except asyncio.TimeoutError:
                pass

            # Feed continuous sensory input
            if vision_encoder.active:
                vis_features = vision_encoder.capture_frame()
                brain.inject_sensory_input("vision", vis_features)
            if audio_encoder.active:
                aud_features = audio_encoder.capture_audio()
                brain.inject_sensory_input("audio", aud_features)

            # Maintain update rate
            elapsed = time.time() - start
            sleep_time = update_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        connected_clients.discard(ws)
        print(f"[WS] Client disconnected ({len(connected_clients)} total)")
