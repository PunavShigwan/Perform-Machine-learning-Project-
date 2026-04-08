"""
app/api/pushup_live_api.py
FastAPI router for live pushup camera session.

Endpoints (prefix /pushup applied in main.py):
  POST  /pushup/live/start        Start webcam session
  POST  /pushup/live/stop         Stop session & get final stats
  GET   /pushup/live/stats        Live stats snapshot
  GET   /pushup/live/stream       MJPEG live video stream
  GET   /pushup/live/health       Health check
"""

import time
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.pushup.pushup_live_service import LivePushupSession

print("📦 pushup_live_api.py — router initialising...")

router = APIRouter()

# One session at a time (module-level singleton)
_session: Optional[LivePushupSession] = None

# =====================================================
# MJPEG GENERATOR
# =====================================================
def _mjpeg_generator():
    global _session
    print("  📡  MJPEG stream — generator started")
    frame_count = 0

    while _session and _session._running:
        frame_bytes = _session.get_frame()
        if frame_bytes is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        if frame_count % 150 == 0:
            print(f"  📡  MJPEG stream — {frame_count} frames delivered")

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps

    print("  📡  MJPEG stream — generator ended (session stopped)")

# =====================================================
# ROUTES
# =====================================================

# ── Health ────────────────────────────────────────
@router.get("/live/health")
def live_health():
    global _session
    active = _session is not None and _session._running
    print(f"  ℹ️   GET /pushup/live/health — session_active={active}")
    return {
        "status":         "ok",
        "session_active": active,
        "stream_url":     "http://localhost:8000/pushup/live/stream",
    }


# ── Start ─────────────────────────────────────────
@router.post("/live/start")
def live_start(camera: int = Query(default=0, description="Webcam index (0 = default)")):
    global _session

    print(f"\n  ▶   POST /pushup/live/start  camera={camera}")

    if _session and _session._running:
        print("  ⚠️   Existing session running — stopping it first")
        _session.stop()

    try:
        _session = LivePushupSession(camera_index=camera)
        _session.start()
        print("  ✅  Live session started successfully")
        return JSONResponse(status_code=200, content={
            "status":       "started",
            "camera_index": camera,
            "stream_url":   "http://localhost:8000/pushup/live/stream",
            "stats_url":    "http://localhost:8000/pushup/live/stats",
            "stop_url":     "http://localhost:8000/pushup/live/stop",
            "message":      (
                "Session started. "
                "Open stream_url in browser or <img> tag to view live feed. "
                "Poll stats_url for live counters."
            ),
        })

    except Exception as e:
        print(f"  ❌  Failed to start session: {e}")
        return JSONResponse(status_code=500, content={
            "status":  "error",
            "message": str(e),
        })


# ── Stop ──────────────────────────────────────────
@router.post("/live/stop")
def live_stop():
    global _session

    print(f"\n  ⏹   POST /pushup/live/stop")

    if _session is None or not _session._running:
        print("  ⚠️   No active session to stop")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /pushup/live/start first.",
        })

    try:
        packet = _session.stop()
        print(f"  ✅  Session stopped — pushups={packet.get('pushup_count')}  "
              f"duration={packet.get('session_duration_sec')}s")
        return JSONResponse(status_code=200, content=packet)

    except Exception as e:
        print(f"  ❌  Error stopping session: {e}")
        return JSONResponse(status_code=500, content={
            "status":  "error",
            "message": str(e),
        })


# ── Live stats ────────────────────────────────────
@router.get("/live/stats")
def live_stats():
    global _session

    print("  ℹ️   GET /pushup/live/stats")

    if _session is None or not _session._running:
        print("  ⚠️   Stats requested but no active session")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /pushup/live/start first.",
        })

    try:
        return JSONResponse(status_code=200, content=_session.get_stats())
    except Exception as e:
        print(f"  ❌  Error getting stats: {e}")
        return JSONResponse(status_code=500, content={
            "status":  "error",
            "message": str(e),
        })


# ── MJPEG stream ──────────────────────────────────
@router.get("/live/stream")
def live_stream():
    global _session

    print("  📡  GET /pushup/live/stream — client connecting")

    if _session is None or not _session._running:
        print("  ⚠️   Stream requested but no active session")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /pushup/live/start first.",
        })

    print("  📡  Streaming MJPEG response...")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

print("✅ pushup_live_api.py — router ready")