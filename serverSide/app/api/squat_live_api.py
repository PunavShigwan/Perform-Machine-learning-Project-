"""
app/api/squat_live_api.py
FastAPI router for live squat webcam session.

Endpoints (prefix /squat applied in main.py):
  POST  /squat/live/start        Start webcam session
  POST  /squat/live/stop         Stop session & get final stats
  GET   /squat/live/stats        Live stats snapshot
  GET   /squat/live/stream       MJPEG live video stream
  GET   /squat/live/health       Health check
"""

import time
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.squat.squat_live_service import LiveSquatSession

print("📦 squat_live_api.py — router initialising...")

router = APIRouter()

# One session at a time (module-level singleton)
_session: Optional[LiveSquatSession] = None


# =====================================================
# MJPEG GENERATOR
# =====================================================
def _mjpeg_generator():
    """Yields annotated JPEG frames as a multipart MJPEG stream."""
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
    print(f"  ℹ️   GET /squat/live/health — session_active={active}")
    return {
        "status":         "ok",
        "session_active": active,
        "stream_url":     "http://localhost:8000/squat/live/stream",
    }


# ── Start ─────────────────────────────────────────
@router.post("/live/start")
def live_start(
    camera: int = Query(default=0, description="Webcam index (0 = default)")
):
    global _session

    print(f"\n  ▶   POST /squat/live/start  camera={camera}")

    if _session and _session._running:
        print("  ⚠️   Existing session running — stopping it first")
        _session.stop()

    try:
        _session = LiveSquatSession(camera_index=camera)
        _session.start()
        print("  ✅  Live squat session started successfully")
        return JSONResponse(status_code=200, content={
            "status":       "started",
            "camera_index": camera,
            "stream_url":   "http://localhost:8000/squat/live/stream",
            "stats_url":    "http://localhost:8000/squat/live/stats",
            "stop_url":     "http://localhost:8000/squat/live/stop",
            "message": (
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

    print(f"\n  ⏹   POST /squat/live/stop")

    if _session is None or not _session._running:
        print("  ⚠️   No active session to stop")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /squat/live/start first.",
        })

    try:
        packet = _session.stop()
        print(
            f"  ✅  Session stopped — "
            f"reps={packet.get('total_reps')}  "
            f"good={packet.get('good_reps')}  "
            f"bad={packet.get('bad_reps')}  "
            f"duration={packet.get('session_duration_sec')}s"
        )
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

    print("  ℹ️   GET /squat/live/stats")

    if _session is None or not _session._running:
        print("  ⚠️   Stats requested but no active session")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /squat/live/start first.",
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

    print("  📡  GET /squat/live/stream — client connecting")

    if _session is None or not _session._running:
        print("  ⚠️   Stream requested but no active session")
        return JSONResponse(status_code=400, content={
            "status":  "error",
            "message": "No active session. Call POST /squat/live/start first.",
        })

    print("  📡  Streaming MJPEG response...")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


print("✅ squat_live_api.py — router ready")