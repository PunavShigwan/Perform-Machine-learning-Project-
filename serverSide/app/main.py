from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

print("=" * 60)
print("  🚀  PERFORM AI Backend  —  Starting up...")
print("=" * 60)

# =====================================================
# OPENAPI TAG DEFINITIONS
# Controls the order and descriptions of tag groups
# displayed in Swagger UI (/docs).
# =====================================================
tags_metadata = [
    {
        "name": "📹 Video Analysis — Pushup",
        "description": (
            "Analyse a **pre-recorded pushup video** file. "
            "Upload a clip and receive rep count, per-rep form scores, "
            "fatigue index, and a processed output video."
        ),
    },
    {
        "name": "🔴 Live Camera — Pushup",
        "description": (
            "Real-time pushup tracking via webcam.\n\n"
            "**Typical flow:**\n"
            "1. `POST /pushup/live/start` — open camera\n"
            "2. `GET  /pushup/live/stream` — embed stream in browser / app\n"
            "3. `GET  /pushup/live/stats`  — poll for live rep & form data\n"
            "4. `POST /pushup/live/stop`   — end session, receive summary"
        ),
    },
    {
        "name": "🏋️ Video Analysis — Clean & Jerk",
        "description": (
            "Analyse a **pre-recorded Clean & Jerk video** file. "
            "Upload a side-on clip and receive phase-by-phase form scores "
            "(clean, jerk, release), a machine call (GREEN / MAJORITY GREEN / RED LIGHT), "
            "and a processed output video."
        ),
    },
    {
        "name": "🏋️ Live Camera — Clean & Jerk",
        "description": (
            "Real-time Clean & Jerk tracking via webcam.\n\n"
            "**Typical flow:**\n"
            "1. `POST /cleanjerk/live/start` — open camera\n"
            "2. `GET  /cleanjerk/live/stream` — embed stream in browser / app\n"
            "3. `GET  /cleanjerk/live/stats`  — poll for live phase & form data\n"
            "4. `POST /cleanjerk/live/stop`   — end session, receive full lift log\n\n"
            "Use `POST /cleanjerk/live/reset` between attempts to commit the current lift "
            "and start fresh without restarting the camera."
        ),
    },
    {
        "name": "💪 Video Analysis — Dips",
        "description": (
            "Analyse a **pre-recorded dips video** file. "
            "Upload a clip and receive rep count, form scores, and a processed output video."
        ),
    },
    {
        "name": "🦵 Video Analysis — Squat",
        "description": (
            "Analyse a **pre-recorded squat video** file. "
            "Upload a clip and receive rep count, good/bad rep breakdown, "
            "per-rep fault log, predicted max reps, and a processed output video."
        ),
    },
    {
        "name": "🔴 Live Camera — Squat",
        "description": (
            "Real-time squat tracking via webcam.\n\n"
            "**Typical flow:**\n"
            "1. `POST /squat/live/start` — open camera\n"
            "2. `GET  /squat/live/stream` — embed stream in browser / app\n"
            "3. `GET  /squat/live/stats`  — poll for live rep & form data\n"
            "4. `POST /squat/live/stop`   — end session, receive summary"
        ),
    },
    {
        "name": "📁 Video Upload",
        "description": "Generic video upload endpoint. Saves the raw file and a processed copy to the output directory.",
    },
]

# =====================================================
# CREATE APP
# =====================================================
app = FastAPI(
    title="PERFORM AI Backend",
    description=(
        "AI-powered exercise analysis API.\n\n"
        "Supports both **pre-recorded video analysis** and **live webcam tracking** "
        "for Pushup, Clean & Jerk, Dips, and Squat.\n\n"
        "| Mode | Exercise | Base path |\n"
        "|------|----------|-----------|\n"
        "| Video analysis | Pushup | `/pushup` |\n"
        "| Video analysis | Clean & Jerk | `/weightlifting` |\n"
        "| Video analysis | Dips | `/dips` |\n"
        "| Video analysis | Squat | `/squat` |\n"
        "| Live camera | Pushup | `/pushup/live` |\n"
        "| Live camera | Clean & Jerk | `/cleanjerk/live` |\n"
        "| Live camera | Squat | `/squat/live` |"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)
print("✅  FastAPI app created")

# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # DEV ONLY — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅  CORS middleware registered")

# =====================================================
# ROUTER IMPORTS
# =====================================================

# ── Pushup (pre-recorded video analysis) ──────────
try:
    print("\n  📦  Importing pushup_api (video analysis)...")
    from app.api.pushup_api import router as pushup_router
    print("  ✅  pushup_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing pushup_api: {e}")
    raise e

# ── Pushup Live (webcam / live camera) ────────────
try:
    print("  📦  Importing pushup_live_api (live camera)...")
    from app.api.pushup_live_api import router as pushup_live_router
    print("  ✅  pushup_live_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing pushup_live_api: {e}")
    raise e

# ── Clean & Jerk (pre-recorded video analysis) ────
try:
    print("  📦  Importing clean_jerk_api...")
    from app.api.clean_jerk_api import router as clean_jerk_router
    print("  ✅  clean_jerk_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing clean_jerk_api: {e}")
    raise e

# ── Clean & Jerk Live (webcam / live camera) ──────
try:
    print("  📦  Importing clean_jerk_live_api (live camera)...")
    from app.api.clean_jerk_live_api import router as clean_jerk_live_router
    print("  ✅  clean_jerk_live_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing clean_jerk_live_api: {e}")
    raise e

# ── Dips (video analysis) ─────────────────────────
try:
    print("  📦  Importing dip_api...")
    from app.api.dip_api import router as dip_router
    print("  ✅  dip_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing dip_api: {e}")
    raise e

# ── Squat (pre-recorded video analysis) ───────────
try:
    print("  📦  Importing squat_api (video analysis)...")
    from app.api.squat_api import router as squat_router
    print("  ✅  squat_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing squat_api: {e}")
    raise e

# ── Squat Live (webcam / live camera) ─────────────
try:
    print("  📦  Importing squat_live_api (live camera)...")
    from app.api.squat_live_api import router as squat_live_router
    print("  ✅  squat_live_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing squat_live_api: {e}")
    raise e

# =====================================================
# REGISTER ROUTERS
# =====================================================
print("\n  🔗  Registering routers...")

# NOTE: Live routers declare their own tags + prefix="/live" internally.
# Video analysis routers get their tags injected here at registration.

app.include_router(
    pushup_router,
    prefix="/pushup",
    tags=["📹 Video Analysis — Pushup"],
)
print("  ✅  /pushup              → pushup_router          (video analysis)")

app.include_router(
    pushup_live_router,
    prefix="/pushup",
    # tag "🔴 Live Camera — Pushup" declared inside the router
    # full paths: /pushup/live/*
)
print("  ✅  /pushup/live/*       → pushup_live_router     (live camera)")

app.include_router(
    clean_jerk_router,
    prefix="/weightlifting",
    tags=["🏋️ Video Analysis — Clean & Jerk"],
)
print("  ✅  /weightlifting       → clean_jerk_router      (video analysis)")

app.include_router(
    clean_jerk_live_router,
    prefix="/cleanjerk",
    # tag "🏋️ Live Camera — Clean & Jerk" declared inside the router
    # full paths: /cleanjerk/live/*
)
print("  ✅  /cleanjerk/live/*    → clean_jerk_live_router (live camera)")

app.include_router(
    dip_router,
    prefix="/dips",
    tags=["💪 Video Analysis — Dips"],
)
print("  ✅  /dips                → dip_router")

app.include_router(
    squat_router,
    prefix="/squat",
    tags=["🦵 Video Analysis — Squat"],
)
print("  ✅  /squat               → squat_router           (video analysis)")

app.include_router(
    squat_live_router,
    prefix="/squat",
    # tag "🔴 Live Camera — Squat" declared inside the router
    # full paths: /squat/live/*
)
print("  ✅  /squat/live/*        → squat_live_router      (live camera)")

# =====================================================
# STATIC FILES  (processed video serving)
# =====================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "uploads", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n  📁  Output dir           : {OUTPUT_DIR}")

app.mount("/local-videos", StaticFiles(directory=OUTPUT_DIR), name="local-videos")
print("  ✅  /local-videos        → StaticFiles (processed videos)")

# =====================================================
# VIDEO UPLOAD + PROCESS ENDPOINT
# =====================================================
@app.post(
    "/upload-video/",
    tags=["📁 Video Upload"],
    summary="Upload a raw video file",
    description=(
        "Saves the uploaded video to the output directory and returns URLs for both "
        "the raw file and a processed copy.\n\n"
        "The processing step is currently a placeholder (file copy). "
        "Replace `shutil.copy` with your ML inference pipeline."
    ),
    response_description="Raw and processed video URLs",
)
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload (mp4, mov, avi, etc.)"),
):
    print(f"\n  📤  POST /upload-video/ — file={file.filename}")

    original_name = file.filename
    uid           = str(uuid.uuid4())
    raw_name      = f"{uid}_{original_name}"
    raw_path      = os.path.join(OUTPUT_DIR, raw_name)

    print(f"       Saving raw upload → {raw_path}")
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"       Raw file saved ({os.path.getsize(raw_path)//1024} KB)")

    # Placeholder processing (replace with ML inference later)
    processed_name = f"processed_{uid}_{original_name}"
    processed_path = os.path.join(OUTPUT_DIR, processed_name)

    shutil.copy(raw_path, processed_path)
    print(f"       Processed copy saved → {processed_path}")

    return {
        "message":             "Video uploaded & processed successfully",
        "original_video_url":  f"http://localhost:8000/local-videos/{raw_name}",
        "processed_video_url": f"http://localhost:8000/local-videos/{processed_name}",
    }

# =====================================================
# ROOT  (excluded from Swagger — use /docs instead)
# =====================================================
@app.get("/", include_in_schema=False)
def root():
    print("  ℹ️   GET / — root endpoint called")
    return {
        "status": "PERFORM AI Backend running",
        "docs":   "http://localhost:8000/docs",
        "redoc":  "http://localhost:8000/redoc",
        "endpoints": {
            # ── Video analysis ──────────────────────────────
            "pushup_video_analyze":    "POST /pushup/analyze",
            "dip_analyze":             "POST /dips/analyze",
            "clean_jerk_analyze":      "POST /weightlifting/clean-jerk/analyze",
            "squat_video_analyze":     "POST /squat/analyze",
            "upload_video":            "POST /upload-video/",
            "processed_videos":        "GET  /local-videos/<filename>",
            # ── Live camera — Pushup ────────────────────────
            "pushup_live_start":       "POST /pushup/live/start?camera=0",
            "pushup_live_stop":        "POST /pushup/live/stop",
            "pushup_live_stats":       "GET  /pushup/live/stats",
            "pushup_live_stream":      "GET  /pushup/live/stream   (MJPEG)",
            "pushup_live_health":      "GET  /pushup/live/health",
            # ── Live camera — Clean & Jerk ──────────────────
            "cleanjerk_live_start":    "POST /cleanjerk/live/start?camera=0",
            "cleanjerk_live_stop":     "POST /cleanjerk/live/stop",
            "cleanjerk_live_reset":    "POST /cleanjerk/live/reset",
            "cleanjerk_live_stats":    "GET  /cleanjerk/live/stats",
            "cleanjerk_live_stream":   "GET  /cleanjerk/live/stream (MJPEG)",
            "cleanjerk_live_health":   "GET  /cleanjerk/live/health",
            # ── Live camera — Squat ─────────────────────────
            "squat_live_start":        "POST /squat/live/start?camera=0",
            "squat_live_stop":         "POST /squat/live/stop",
            "squat_live_stats":        "GET  /squat/live/stats",
            "squat_live_stream":       "GET  /squat/live/stream     (MJPEG)",
            "squat_live_health":       "GET  /squat/live/health",
            # ── Docs ────────────────────────────────────────
            "swagger_ui":              "GET  /docs",
            "redoc":                   "GET  /redoc",
        }
    }

# =====================================================
# STARTUP / SHUTDOWN EVENTS
# =====================================================
@app.on_event("startup")
async def on_startup():
    print("\n" + "=" * 60)
    print("  ✅  PERFORM AI Backend is READY")
    print("=" * 60)
    print("  Base URL        : http://localhost:8000")
    print("  Swagger UI      : http://localhost:8000/docs")
    print("  ReDoc           : http://localhost:8000/redoc")
    print()
    print("  VIDEO ANALYSIS")
    print("    POST  http://localhost:8000/pushup/analyze")
    print("    POST  http://localhost:8000/dips/analyze")
    print("    POST  http://localhost:8000/weightlifting/clean-jerk/analyze")
    print("    POST  http://localhost:8000/squat/analyze")
    print("    POST  http://localhost:8000/upload-video/")
    print("    GET   http://localhost:8000/local-videos/<filename>")
    print()
    print("  LIVE CAMERA — PUSHUP")
    print("    POST  http://localhost:8000/pushup/live/start?camera=0")
    print("    POST  http://localhost:8000/pushup/live/stop")
    print("    GET   http://localhost:8000/pushup/live/stats")
    print("    GET   http://localhost:8000/pushup/live/stream")
    print("    GET   http://localhost:8000/pushup/live/health")
    print()
    print("  LIVE CAMERA — CLEAN & JERK")
    print("    POST  http://localhost:8000/cleanjerk/live/start?camera=0")
    print("    POST  http://localhost:8000/cleanjerk/live/stop")
    print("    POST  http://localhost:8000/cleanjerk/live/reset")
    print("    GET   http://localhost:8000/cleanjerk/live/stats")
    print("    GET   http://localhost:8000/cleanjerk/live/stream")
    print("    GET   http://localhost:8000/cleanjerk/live/health")
    print()
    print("  LIVE CAMERA — SQUAT")
    print("    POST  http://localhost:8000/squat/live/start?camera=0")
    print("    POST  http://localhost:8000/squat/live/stop")
    print("    GET   http://localhost:8000/squat/live/stats")
    print("    GET   http://localhost:8000/squat/live/stream")
    print("    GET   http://localhost:8000/squat/live/health")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def on_shutdown():
    print("\n  ⏹   PERFORM AI Backend shutting down...")

    # ── Stop pushup live session ───────────────────────────────
    try:
        from app.api.pushup_live_api import _session as _pushup_session
        if _pushup_session and _pushup_session._running:
            print("  ⏹   Stopping active pushup live session...")
            _pushup_session.stop()
            print("  ✅  Pushup live session stopped cleanly")
    except Exception as e:
        print(f"  ⚠️   Could not stop pushup live session on shutdown: {e}")

    # ── Stop clean & jerk live session ────────────────────────
    try:
        from app.api.clean_jerk_live_api import _session as _cj_session
        if _cj_session and _cj_session._running:
            print("  ⏹   Stopping active C&J live session...")
            _cj_session.stop()
            print("  ✅  C&J live session stopped cleanly")
    except Exception as e:
        print(f"  ⚠️   Could not stop C&J live session on shutdown: {e}")

    # ── Stop squat live session ────────────────────────────────
    try:
        from app.api.squat_live_api import _session as _squat_session
        if _squat_session and _squat_session._running:
            print("  ⏹   Stopping active squat live session...")
            _squat_session.stop()
            print("  ✅  Squat live session stopped cleanly")
    except Exception as e:
        print(f"  ⚠️   Could not stop squat live session on shutdown: {e}")

    print("  👋  Goodbye!\n")


print("\n✅  main.py loaded — all routers registered")