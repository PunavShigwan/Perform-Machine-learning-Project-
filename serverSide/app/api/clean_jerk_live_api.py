"""
app/api/clean_jerk_live_api.py
FastAPI router for live clean & jerk camera session.

Endpoints (prefix /cleanjerk applied in main.py):
  POST  /cleanjerk/live/start        Start webcam session
  POST  /cleanjerk/live/stop         Stop session & get final stats
  POST  /cleanjerk/live/reset        Force-commit current lift & begin a new one
  GET   /cleanjerk/live/stats        Live stats snapshot
  GET   /cleanjerk/live/stream       MJPEG live video stream
  GET   /cleanjerk/live/health       Health check

HOW A LIFT IS TRACKED
─────────────────────
One session can contain multiple lifts.  A lift starts when a bar-touch is
detected (wrists at knee level, close together) and ends automatically after
IDLE_RESET_FRAMES (~3 s) of inactivity once RELEASE_FINISH has been detected.

You can also call POST /cleanjerk/live/reset to manually end the current lift
attempt and immediately begin tracking the next one — useful for training
sessions with multiple attempts.
"""

import time
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.services.weightlifting.clean_jerk.clean_jerk_live_service    import LiveCleanJerkSession

print("📦 clean_jerk_live_api.py — router initialising...")

router = APIRouter()

# One session at a time (module-level singleton)
_session: Optional[LiveCleanJerkSession] = None


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
            + frame_bytes
            + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps cap

    print("  📡  MJPEG stream — generator ended (session stopped)")


# =====================================================
# ROUTES
# =====================================================

# ── Health ────────────────────────────────────────────────────
@router.get("/live/health")
def live_health():
    global _session
    active    = _session is not None and _session._running
    lift_cnt  = _session.lift_count if active else 0
    stage     = _session.current_stage if active else None
    print(f"  ℹ️   GET /cleanjerk/live/health — active={active}")
    return {
        "status":         "ok",
        "session_active": active,
        "lift_count":     lift_cnt,
        "current_stage":  stage,
        "stream_url":     "http://localhost:8000/cleanjerk/live/stream",
        "stats_url":      "http://localhost:8000/cleanjerk/live/stats",
    }


# ── Start ─────────────────────────────────────────────────────
@router.post("/live/start")
def live_start(
    camera: int = Query(
        default=0,
        description="Webcam index (0 = default system camera)",
    )
):
    global _session
    print(f"\n  ▶   POST /cleanjerk/live/start  camera={camera}")

    if _session and _session._running:
        print("  ⚠️   Existing session running — stopping it first")
        _session.stop()

    try:
        _session = LiveCleanJerkSession(camera_index=camera)
        _session.start()
        print("  ✅  Live C&J session started successfully")
        return JSONResponse(
            status_code=200,
            content={
                "status":       "started",
                "camera_index": camera,
                "stream_url":   "http://localhost:8000/cleanjerk/live/stream",
                "stats_url":    "http://localhost:8000/cleanjerk/live/stats",
                "stop_url":     "http://localhost:8000/cleanjerk/live/stop",
                "reset_url":    "http://localhost:8000/cleanjerk/live/reset",
                "message": (
                    "Session started. "
                    "Open stream_url in browser or <img> tag to view live feed. "
                    "Poll stats_url for live lift data. "
                    "A lift begins automatically when bar-touch is detected; "
                    "call reset_url to manually commit the current attempt and begin the next."
                ),
            },
        )
    except Exception as e:
        print(f"  ❌  Failed to start session: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── Stop ──────────────────────────────────────────────────────
@router.post("/live/stop")
def live_stop():
    global _session
    print(f"\n  ⏹   POST /cleanjerk/live/stop")

    if _session is None or not _session._running:
        print("  ⚠️   No active session to stop")
        return JSONResponse(
            status_code=400,
            content={
                "status":  "error",
                "message": "No active session. Call POST /cleanjerk/live/start first.",
            },
        )

    try:
        packet = _session.stop()
        print(
            f"  ✅  Session stopped — lifts={packet.get('lift_count')}  "
            f"duration={packet.get('session_duration_sec')}s"
        )
        return JSONResponse(status_code=200, content=packet)
    except Exception as e:
        print(f"  ❌  Error stopping session: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── Reset (force-commit current lift, start fresh) ────────────
@router.post("/live/reset")
def live_reset():
    """
    Manually end the current lift attempt and prepare for the next.
    Useful when the athlete wants to re-attempt before the auto-idle timeout.

    The background thread's lift state is reset by injecting a sentinel;
    the session itself (camera stream) keeps running uninterrupted.
    """
    global _session
    print(f"\n  🔄  POST /cleanjerk/live/reset")

    if _session is None or not _session._running:
        return JSONResponse(
            status_code=400,
            content={
                "status":  "error",
                "message": "No active session. Call POST /cleanjerk/live/start first.",
            },
        )

    try:
        # Signal the background thread to commit + reset on its next tick.
        # We set a flag the loop checks; no direct thread manipulation needed.
        _session._force_reset = True

        return JSONResponse(
            status_code=200,
            content={
                "status":     "reset_requested",
                "lift_count": _session.lift_count,
                "message":    (
                    "Current lift will be committed and a new lift will begin "
                    "on the next bar-touch detection."
                ),
            },
        )
    except Exception as e:
        print(f"  ❌  Error requesting reset: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── Live stats ────────────────────────────────────────────────
@router.get("/live/stats")
def live_stats():
    global _session
    print("  ℹ️   GET /cleanjerk/live/stats")

    if _session is None or not _session._running:
        return JSONResponse(
            status_code=400,
            content={
                "status":  "error",
                "message": "No active session. Call POST /cleanjerk/live/start first.",
            },
        )

    try:
        return JSONResponse(status_code=200, content=_session.get_stats())
    except Exception as e:
        print(f"  ❌  Error getting stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── MJPEG stream ──────────────────────────────────────────────
@router.get("/live/stream")
def live_stream():
    global _session
    print("  📡  GET /cleanjerk/live/stream — client connecting")

    if _session is None or not _session._running:
        return JSONResponse(
            status_code=400,
            content={
                "status":  "error",
                "message": "No active session. Call POST /cleanjerk/live/start first.",
            },
        )

    print("  📡  Streaming MJPEG response...")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


print("✅ clean_jerk_live_api.py — router ready")