"""
FastAPI + WebSocket Server for NeuroLinked Brain

Serves the 3D dashboard and streams real-time brain state via WebSocket.
Provides Claude integration API for reading brain state and sending input.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("brainflow")

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

# Security: Use configured CORS origins instead of wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=BrainConfig.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    allow_credentials=True,
)

# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Content Security Policy
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws: wss:;"
    # Strict Transport Security (only in production)
    if BrainConfig.is_production():
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Global auth middleware - protects ALL routes except allow-listed ones
@app.middleware("http")
async def global_auth(request: Request, call_next):
    """
    Global authentication middleware.
    
    All routes require authentication EXCEPT:
    - / (index page)
    - /css/*, /js/* (static assets)
    - /api/health (health check)
    - /api/version (version info)
    - /metrics (prometheus metrics)
    
    WebSocket /ws is handled separately in the WS handler.
    """
    if not BrainConfig.REQUIRE_AUTH:
        return await call_next(request)
    
    path = request.url.path
    
    # Allow-list: paths that don't require authentication
    PUBLIC_PATHS = [
        "/",
        "/api/health",
        "/api/version",
        "/metrics",
    ]
    
    # Static assets
    if path.startswith("/css/") or path.startswith("/js/"):
        return await call_next(request)
    
    # Check if path is public
    if path in PUBLIC_PATHS:
        return await call_next(request)
    
    # WebSocket is handled separately
    if path == "/ws":
        return await call_next(request)
    
    # Check for Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {"error": "Authentication required", "detail": "Missing or invalid Authorization header"},
            status_code=401
        )
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    if not BrainConfig.validate_token(token):
        return JSONResponse(
            {"error": "Authentication failed", "detail": "Invalid or expired token"},
            status_code=403
        )
    
    return await call_next(request)

# Auth setup
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token for protected endpoints."""
    if not BrainConfig.REQUIRE_AUTH:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    token = credentials.credentials
    if not BrainConfig.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid or expired token")
    
    return True

# Rate limiting storage (simple in-memory, or Redis for distributed)
_rate_limit_store = {}
_redis_client = None

def _get_redis():
    """Get Redis client if configured."""
    global _redis_client
    if _redis_client is None and BrainConfig.REDIS_URL:
        try:
            import redis
            _redis_client = redis.from_url(BrainConfig.REDIS_URL, decode_responses=True)
            logger.info("Redis rate limiting enabled")
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to in-memory: {e}")
            _redis_client = False
    return _redis_client if _redis_client else None

async def rate_limit(request: Request):
    """Rate limiting middleware with Redis support for distributed deployments."""
    if not BrainConfig.RATE_LIMIT_ENABLED:
        return True
    
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_ip}"
    
    # Try Redis first (for distributed rate limiting)
    redis_client = _get_redis()
    if redis_client:
        try:
            pipe = redis_client.pipeline()
            now = time.time()
            window_start = now - BrainConfig.RATE_LIMIT_WINDOW
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Count requests in window
            pipe.zcard(key)
            # Set expiry on key
            pipe.expire(key, BrainConfig.RATE_LIMIT_WINDOW)
            
            results = pipe.execute()
            count = results[2]
            
            if count > BrainConfig.RATE_LIMIT_REQUESTS:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            return True
        except Exception as e:
            logger.warning(f"Redis rate limit failed, falling back to memory: {e}")
            # Fall through to in-memory
    
    # In-memory rate limiting (per-process only)
    now = time.time()
    
    # Clean old entries
    for ip in list(_rate_limit_store.keys()):
        if now - _rate_limit_store[ip]["window_start"] > BrainConfig.RATE_LIMIT_WINDOW:
            del _rate_limit_store[ip]
    
    # Check rate limit
    if client_ip in _rate_limit_store:
        data = _rate_limit_store[client_ip]
        if now - data["window_start"] > BrainConfig.RATE_LIMIT_WINDOW:
            # New window
            _rate_limit_store[client_ip] = {"count": 1, "window_start": now}
        else:
            data["count"] += 1
            if data["count"] > BrainConfig.RATE_LIMIT_REQUESTS:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        _rate_limit_store[client_ip] = {"count": 1, "window_start": now}
    
    return True

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

# WebSocket rate limiting - per-IP reconnect tracking
_ws_reconnect_tracker = {}
_WS_RECONNECT_LIMIT = 10  # Max reconnects per window
_WS_RECONNECT_WINDOW = 60  # Window in seconds

# Brain access lock - protects brain state from race conditions
# HTTP/WS handlers acquire this; sim thread runs independently
brain_lock = threading.RLock()

# Async-safe state snapshot for WebSocket broadcasting
_latest_state_snapshot = None
_snapshot_lock = asyncio.Lock()

# Auto-save interval (every 5 minutes)
AUTO_SAVE_INTERVAL = 300
_last_auto_save = 0


def _build_state_snapshot():
    """
    Build a complete state snapshot from brain and related systems.
    
    This is called from the simulation thread while holding brain_lock.
    All brain state reads happen here.
    """
    snapshot = {
        "timestamp": time.time(),
        "brain": None,
        "claude": None,
        "screen_observer": None,
        "video_recorder": None,
    }
    
    if brain:
        snapshot["brain"] = brain.get_state()
    
    if claude_bridge:
        snapshot["claude"] = {
            "connected": True,
            "interactions": claude_bridge._interaction_count,
        }
    
    if screen_observer:
        snapshot["screen_observer"] = screen_observer.get_state()
    
    if video_recorder:
        snapshot["video_recorder"] = video_recorder.get_state()
    
    return snapshot


async def _publish_snapshot(snapshot):
    """
    Publish state snapshot for HTTP/WS consumers.
    
    Called from simulation thread via run_coroutine_threadsafe.
    """
    global _latest_state_snapshot
    async with _snapshot_lock:
        _latest_state_snapshot = snapshot


async def get_latest_snapshot():
    """Get the latest state snapshot (for HTTP/WS handlers)."""
    async with _snapshot_lock:
        return _latest_state_snapshot


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
_last_knowledge_prune = 0
KNOWLEDGE_PRUNE_INTERVAL = 3600  # Prune knowledge every hour

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
                except Exception as e:
                    logger.error(f"Screen logging failed: {e}", exc_info=True)
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
                except Exception as e:
                    logger.error(f"Brain event logging failed: {e}", exc_info=True)
                _last_brain_event_log = now

            # Step the brain with lock protection
            with brain_lock:
                brain.step()
                
                # PUBLISH SNAPSHOT: Build state snapshot while holding lock
                # This is the ONLY place that reads brain state
                snapshot = _build_state_snapshot()
            
            # Publish snapshot for HTTP/WS consumers (outside lock)
            asyncio.run_coroutine_threadsafe(
                _publish_snapshot(snapshot), 
                asyncio.get_event_loop()
            )

            # Auto-save periodically (with lock)
            now = time.time()
            if now - _last_auto_save > AUTO_SAVE_INTERVAL:
                try:
                    with brain_lock:
                        save_brain(brain)
                    _last_auto_save = now
                    logger.info(f"Auto-saved brain at step {brain.step_count}")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}", exc_info=True)
            
            # P1: Prune knowledge store periodically
            global _last_knowledge_prune
            if now - _last_knowledge_prune > KNOWLEDGE_PRUNE_INTERVAL:
                try:
                    if claude_bridge and claude_bridge.knowledge:
                        deleted = claude_bridge.knowledge.prune_old_entries()
                        if deleted > 100:  # Vacuum if significant pruning
                            claude_bridge.knowledge.vacuum()
                        _last_knowledge_prune = now
                        logger.info(f"Pruned {deleted} old knowledge entries")
                except Exception as e:
                    logger.error(f"Knowledge pruning failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Simulation loop error: {e}", exc_info=True)
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
    logger.info("Simulation started")


def stop_simulation():
    """Stop the background simulation thread."""
    global sim_running
    sim_running = False
    logger.info("Simulation stop requested")
    
    # Wait for thread to finish with timeout
    if sim_thread and sim_thread.is_alive():
        sim_thread.join(timeout=5.0)
        if sim_thread.is_alive():
            logger.warning("Simulation thread did not stop within timeout")
        else:
            logger.info("Simulation stopped cleanly")


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
    logger.info("Shutdown initiated...")
    
    # Stop simulation first
    stop_simulation()
    
    # Save brain state on shutdown (with lock protection)
    try:
        if brain:
            with brain_lock:
                save_brain(brain)
            logger.info("Brain saved on shutdown")
    except Exception as e:
        logger.error(f"Save on shutdown failed: {e}", exc_info=True)

    # Stop all encoders/observers
    if screen_observer:
        try:
            screen_observer.stop()
            logger.info("Screen observer stopped")
        except Exception as e:
            logger.error(f"Screen observer stop failed: {e}")
    if video_recorder:
        try:
            video_recorder.stop()
            logger.info("Video recorder stopped")
        except Exception as e:
            logger.error(f"Video recorder stop failed: {e}")
    if vision_encoder:
        try:
            vision_encoder.stop_webcam()
            logger.info("Vision encoder stopped")
        except Exception as e:
            logger.error(f"Vision encoder stop failed: {e}")
    if audio_encoder:
        try:
            audio_encoder.stop_microphone()
            logger.info("Audio encoder stopped")
        except Exception as e:
            logger.error(f"Audio encoder stop failed: {e}")
    
    logger.info("Shutdown complete")


# =============================================================================
# Metrics Endpoint (P1: Prometheus metrics)
# =============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint - no authentication required."""
    if not BrainConfig.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    lines = []
    
    # Brain metrics
    if brain:
        lines.append(f"# HELP brainflow_steps_total Total brain steps")
        lines.append(f"# TYPE brainflow_steps_total counter")
        lines.append(f"brainflow_steps_total {brain.step_count}")
        
        lines.append(f"# HELP brainflow_steps_per_second Current simulation rate")
        lines.append(f"# TYPE brainflow_steps_per_second gauge")
        lines.append(f"brainflow_steps_per_second {getattr(brain, 'steps_per_second', 0)}")
        
        lines.append(f"# HELP brainflow_neuromodulator_level Neuromodulator levels")
        lines.append(f"# TYPE brainflow_neuromodulator_level gauge")
        for nm, val in brain.neuromodulators.items():
            lines.append(f'brainflow_neuromodulator_level{{type="{nm}"}} {val}')
    
    # Connection metrics
    lines.append(f"# HELP brainflow_connected_clients Number of connected WebSocket clients")
    lines.append(f"# TYPE brainflow_connected_clients gauge")
    lines.append(f"brainflow_connected_clients {len(connected_clients)}")
    
    # Simulation state
    lines.append(f"# HELP brainflow_simulation_running Is simulation running")
    lines.append(f"# TYPE brainflow_simulation_running gauge")
    lines.append(f"brainflow_simulation_running {1 if sim_running else 0}")
    
    return "\n".join(lines)


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse(os.path.join(dashboard_path, "index.html"))


# =============================================================================
# Public Health & Version Endpoints (No Auth Required)
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint - no authentication required."""
    health = {
        "status": "healthy",
        "brain_initialized": brain is not None,
        "simulation_running": sim_running,
        "timestamp": time.time(),
    }
    if brain:
        health["brain"] = {
            "step_count": brain.step_count,
            "stage": brain.development_stage,
            "steps_per_second": getattr(brain, "steps_per_second", 0),
        }
    return JSONResponse(health)


@app.get("/api/version")
async def version():
    """Version info endpoint - no authentication required."""
    return JSONResponse({
        "version": "1.0.0",
        "schema_version": "1.0.0",
        "api_version": "v1",
        "build": os.environ.get("NEUROLINKED_BUILD", "dev"),
        "environment": "production" if BrainConfig.is_production() else "development",
    })


# =============================================================================
# Dashboard API (Protected - Requires Auth via global middleware)
# =============================================================================

@app.get("/api/state", dependencies=[Depends(rate_limit)])
async def get_state():
    """Get current brain state from snapshot (no lock needed)."""
    try:
        snapshot = await get_latest_snapshot()
        if snapshot and snapshot.get("brain"):
            return JSONResponse(snapshot["brain"])
        # Fallback: return minimal state if no snapshot yet
        return JSONResponse({"status": "initializing"})
    except Exception as e:
        logger.error(f"Failed to get state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get brain state")


@app.get("/api/positions", dependencies=[Depends(rate_limit)])
async def get_positions():
    """Get neuron positions for 3D visualization."""
    try:
        with brain_lock:
            positions = brain.get_neuron_positions()
        return JSONResponse(positions)
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get neuron positions")


@app.post("/api/input/text", dependencies=[Depends(rate_limit)])
async def input_text(data: dict):
    """Inject text input into the brain."""
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        features = text_encoder.encode(text)
        with brain_lock:
            brain.inject_sensory_input("text", features)
        # Also log to Claude bridge
        if claude_bridge:
            claude_bridge.send_observation({
                "type": "text",
                "content": text,
                "source": "user",
            })
        return {"status": "ok", "encoded_dim": len(features)}
    except Exception as e:
        logger.error(f"Text input failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process text input")


@app.post("/api/input/vision/start", dependencies=[Depends(rate_limit)])
async def start_vision():
    """Start webcam vision input."""
    try:
        success = vision_encoder.start_webcam()
        logger.info(f"Vision encoder start: {success}")
        return {"status": "started" if success else "unavailable"}
    except Exception as e:
        logger.error(f"Vision start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start vision")


@app.post("/api/input/vision/stop", dependencies=[Depends(rate_limit)])
async def stop_vision():
    """Stop webcam vision input."""
    try:
        vision_encoder.stop_webcam()
        logger.info("Vision encoder stopped")
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Vision stop failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to stop vision")


@app.post("/api/input/audio/start", dependencies=[Depends(rate_limit)])
async def start_audio():
    """Start microphone audio input."""
    try:
        success = audio_encoder.start_microphone()
        logger.info(f"Audio encoder start: {success}")
        return {"status": "started" if success else "unavailable"}
    except Exception as e:
        logger.error(f"Audio start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start audio")


@app.post("/api/input/audio/stop", dependencies=[Depends(rate_limit)])
async def stop_audio():
    """Stop microphone audio input."""
    try:
        audio_encoder.stop_microphone()
        logger.info("Audio encoder stopped")
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Audio stop failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to stop audio")


@app.post("/api/control/pause", dependencies=[Depends(rate_limit)])
async def pause():
    """Pause brain simulation."""
    try:
        stop_simulation()
        logger.info("Simulation paused")
        return {"status": "paused"}
    except Exception as e:
        logger.error(f"Pause failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to pause simulation")


@app.post("/api/control/resume", dependencies=[Depends(rate_limit)])
async def resume():
    """Resume brain simulation."""
    try:
        start_simulation()
        logger.info("Simulation resumed")
        return {"status": "running"}
    except Exception as e:
        logger.error(f"Resume failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resume simulation")


@app.post("/api/control/reset", dependencies=[Depends(rate_limit)])
async def reset():
    """Reset brain to initial state."""
    try:
        stop_simulation()
        init_brain()
        start_simulation()
        logger.info("Brain reset")
        return {"status": "reset"}
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset brain")


# =============================================================================
# Claude Integration API (Protected - Requires Auth)
# =============================================================================

@app.get("/api/claude/summary", dependencies=[Depends(rate_limit)])
async def claude_summary():
    """Primary endpoint for Claude to read brain state."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_brain_summary()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/insights", dependencies=[Depends(rate_limit)])
async def claude_insights():
    """Get brain-derived insights useful for Claude."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_insights()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/claude/observe", dependencies=[Depends(rate_limit)])
async def claude_observe(data: dict):
    """
    Claude sends an observation to the brain.
    Body: {"type": "text"|"action"|"context", "content": "...", "source": "claude"}
    """
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    claude_bridge.send_observation(data)
    return {"status": "ok", "interaction_count": claude_bridge._interaction_count}


@app.get("/api/claude/status", dependencies=[Depends(rate_limit)])
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


@app.get("/api/claude/activity", dependencies=[Depends(rate_limit)])
async def claude_activity():
    """Get recent activity log."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    return JSONResponse(claude_bridge.get_activity_log())


@app.get("/api/claude/learned", dependencies=[Depends(rate_limit)])
async def claude_learned():
    """Get what the brain has learned - grouped patterns and associations."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_learned_patterns()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/learned/summary", dependencies=[Depends(rate_limit)])
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

@app.get("/api/claude/recall", dependencies=[Depends(rate_limit)])
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


@app.get("/api/claude/search", dependencies=[Depends(rate_limit)])
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


@app.get("/api/claude/semantic", dependencies=[Depends(rate_limit)])
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


@app.get("/api/claude/knowledge", dependencies=[Depends(rate_limit)])
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


@app.post("/api/claude/remember", dependencies=[Depends(rate_limit)])
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
# WebSocket for real-time streaming (AUTHENTICATED)
# =============================================================================

async def verify_ws_token(ws: WebSocket) -> bool:
    """Verify authentication token from WebSocket query params or first message."""
    if not BrainConfig.REQUIRE_AUTH:
        return True
    
    # Try to get token from query parameters
    token = ws.query_params.get("token", "")
    
    if not token:
        # Try to receive first message as auth handshake
        try:
            auth_msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            if auth_msg.get("type") == "auth":
                token = auth_msg.get("token", "")
            else:
                logger.warning("WS: First message was not auth handshake")
                return False
        except asyncio.TimeoutError:
            logger.warning("WS: Auth handshake timeout")
            return False
        except Exception as e:
            logger.warning(f"WS: Auth handshake error: {e}")
            return False
    
    if not BrainConfig.validate_token(token):
        logger.warning("WS: Invalid token rejected")
        return False
    
    return True


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # P1: Rate limit WS reconnects per IP
    client_ip = ws.client.host if ws.client else "unknown"
    now = time.time()
    
    # Clean old entries
    for ip in list(_ws_reconnect_tracker.keys()):
        if now - _ws_reconnect_tracker[ip]["window_start"] > _WS_RECONNECT_WINDOW:
            del _ws_reconnect_tracker[ip]
    
    # Check reconnect limit
    if client_ip in _ws_reconnect_tracker:
        data = _ws_reconnect_tracker[client_ip]
        if now - data["window_start"] > _WS_RECONNECT_WINDOW:
            # New window
            _ws_reconnect_tracker[client_ip] = {"count": 1, "window_start": now}
        else:
            data["count"] += 1
            if data["count"] > _WS_RECONNECT_LIMIT:
                await ws.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Reconnect rate limit exceeded")
                logger.warning(f"WS: Reconnect rate limit exceeded for {client_ip}")
                return
    else:
        _ws_reconnect_tracker[client_ip] = {"count": 1, "window_start": now}
    
    # Authenticate before accepting
    if not await verify_ws_token(ws):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
        logger.warning("WS: Connection rejected - authentication failed")
        return
    
    await ws.accept()
    connected_clients.add(ws)
    logger.info(f"WS: Client connected ({len(connected_clients)} total)")

    # Send initial neuron positions (with lock protection)
    try:
        with brain_lock:
            positions = brain.get_neuron_positions()
        await ws.send_json({"type": "init", "positions": positions})
    except Exception as e:
        logger.error(f"WS: Failed to send initial positions: {e}", exc_info=True)

    try:
        update_interval = 1.0 / BrainConfig.WS_UPDATE_RATE
        last_state_hash = None
        
        while True:
            start = time.time()

            # Get state snapshot (published by sim thread, no lock needed)
            try:
                snapshot = await get_latest_snapshot()
                if snapshot and snapshot.get("brain"):
                    state = snapshot["brain"].copy()
                    
                    # Add Claude bridge info from snapshot
                    if snapshot.get("claude"):
                        state["claude"] = snapshot["claude"]
                    if snapshot.get("screen_observer"):
                        state["screen_observer"] = snapshot["screen_observer"]
                    if snapshot.get("video_recorder"):
                        state["video_recorder"] = snapshot["video_recorder"]
                    
                    # Only send if state changed (simple hash check)
                    state_hash = hash(json.dumps(state, sort_keys=True, default=str))
                    if state_hash != last_state_hash:
                        await ws.send_json({"type": "state", "data": state})
                        last_state_hash = state_hash
                    
            except Exception as e:
                logger.error(f"WS: Failed to get/send state: {e}", exc_info=True)

            # Check for incoming messages (text input, commands)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                
                # Handle auth message (already authenticated, but allow re-auth)
                if msg.get("type") == "auth":
                    continue
                    
                if msg.get("type") == "text_input":
                    text = msg.get("text", "")
                    if text:
                        features = text_encoder.encode(text)
                        with brain_lock:
                            brain.inject_sensory_input("text", features)
                        if claude_bridge:
                            claude_bridge.send_observation({
                                "type": "text",
                                "content": text,
                                "source": "dashboard",
                            })
                            
                elif msg.get("type") == "command":
                    cmd = msg.get("cmd")
                    logger.info(f"WS: Received command: {cmd}")
                    
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
                        with brain_lock:
                            save_brain(brain)
                        logger.info("WS: Brain saved via command")
                    elif cmd == "load":
                        with brain_lock:
                            load_brain(brain)
                        logger.info("WS: Brain loaded via command")
                    else:
                        logger.warning(f"WS: Unknown command: {cmd}")
                        
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"WS: Error processing message: {e}", exc_info=True)

            # Feed continuous sensory input (with lock protection)
            try:
                if vision_encoder.active:
                    vis_features = vision_encoder.capture_frame()
                    with brain_lock:
                        brain.inject_sensory_input("vision", vis_features)
                if audio_encoder.active:
                    aud_features = audio_encoder.capture_audio()
                    with brain_lock:
                        brain.inject_sensory_input("audio", aud_features)
            except Exception as e:
                logger.error(f"WS: Sensory input error: {e}", exc_info=True)

            # Maintain update rate
            elapsed = time.time() - start
            sleep_time = update_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("WS: Client disconnected normally")
    except Exception as e:
        logger.error(f"WS: Unexpected error: {e}", exc_info=True)
    finally:
        connected_clients.discard(ws)
        logger.info(f"WS: Client disconnected ({len(connected_clients)} total)")
