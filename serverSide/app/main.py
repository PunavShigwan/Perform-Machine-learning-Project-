<<<<<<< HEAD
=======

>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

<<<<<<< HEAD
print("🚀 Starting FastAPI application...")
=======
print("=" * 60)
print("  🚀  PERFORM AI Backend  —  Starting up...")
print("=" * 60)
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

# =====================================================
# CREATE APP
# =====================================================
<<<<<<< HEAD
app = FastAPI(title="PERFORM AI Backend")
=======
app = FastAPI(
    title="PERFORM AI Backend",
    description="AI-powered exercise analysis API",
    version="1.0.0",
)
print("✅  FastAPI app created")
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
<<<<<<< HEAD
    allow_origins=["*"],        # DEV ONLY
=======
    allow_origins=["*"],        # DEV ONLY — restrict in production
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅  CORS middleware registered")

# =====================================================
<<<<<<< HEAD
# ROUTES (PREFIX ONLY HERE)
# =====================================================
try:
    from app.api.pushup_api import router as pushup_router
    print("📦 Loading pushup_api...")
    print("✅ pushup_api imported successfully")
except Exception as e:
    print("❌ ERROR importing pushup_api:", e)
    raise e

try:
    from app.api.clean_jerk_api import router as clean_jerk_router
    print("📦 Loading clean_jerk_api...")
    print("✅ clean_jerk_api imported successfully")
except Exception as e:
    print("❌ ERROR importing clean_jerk_api:", e)
    raise e

# ✅ PREFIXES APPLIED HERE
app.include_router(pushup_router, prefix="/pushup")
app.include_router(clean_jerk_router, prefix="/weightlifting")

# =====================================================
# STATIC FILES (PROCESSED VIDEO SERVING)
# =====================================================

# BASE_DIR = C:\major_project\serverSide\app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This MUST match where your processed videos are saved:
# C:\major_project\serverSide\app\uploads\output
OUTPUT_DIR = os.path.join(BASE_DIR, "uploads", "output")

# Ensure folder exists (prevents silent issues)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🎥 Serving processed videos from:", OUTPUT_DIR)

# Public URL: http://localhost:8000/local-videos/<filename>
app.mount("/local-videos", StaticFiles(directory=OUTPUT_DIR), name="local-videos")
=======
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

# ── Clean & Jerk ──────────────────────────────────
try:
    print("  📦  Importing clean_jerk_api...")
    from app.api.clean_jerk_api import router as clean_jerk_router
    print("  ✅  clean_jerk_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing clean_jerk_api: {e}")
    raise e

# ── Dips (video analysis) ─────────────────────────
try:
    print("  📦  Importing dip_api...")
    from app.api.dip_api import router as dip_router
    print("  ✅  dip_api imported successfully")
except Exception as e:
    print(f"  ❌  ERROR importing dip_api: {e}")
    raise e

# =====================================================
# REGISTER ROUTERS
# =====================================================
print("\n  🔗  Registering routers...")

app.include_router(pushup_router, prefix="/pushup")
print("  ✅  /pushup            → pushup_router      (video analysis)")

app.include_router(pushup_live_router, prefix="/pushup")
print("  ✅  /pushup/live/*     → pushup_live_router (live camera)")

app.include_router(clean_jerk_router, prefix="/weightlifting")
print("  ✅  /weightlifting     → clean_jerk_router")

app.include_router(dip_router, prefix="/dips")
print("  ✅  /dips              → dip_router")

# =====================================================
# STATIC FILES  (processed video serving)
# =====================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "uploads", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n  📁  Output dir         : {OUTPUT_DIR}")

app.mount("/local-videos", StaticFiles(directory=OUTPUT_DIR), name="local-videos")
print("  ✅  /local-videos      → StaticFiles (processed videos)")
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69

# =====================================================
# VIDEO UPLOAD + PROCESS ENDPOINT
# =====================================================
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
<<<<<<< HEAD
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]

    uid = str(uuid.uuid4())
    raw_name = f"{uid}_{original_name}"
    raw_path = os.path.join(OUTPUT_DIR, raw_name)

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 👉 Hook your real ML/video processing here
    processed_name = f"processed_{uid}_{original_name}"
    processed_path = os.path.join(OUTPUT_DIR, processed_name)

    # TEMP: just copy original as processed
    shutil.copy(raw_path, processed_path)

    return {
        "message": "Video uploaded & processed successfully",
        "original_video_url": f"http://localhost:8000/local-videos/{raw_name}",
        "processed_video_url": f"http://localhost:8000/local-videos/{processed_name}"
=======
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
        "message": "Video uploaded & processed successfully",
        "original_video_url":  f"http://localhost:8000/local-videos/{raw_name}",
        "processed_video_url": f"http://localhost:8000/local-videos/{processed_name}",
>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
    }

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
<<<<<<< HEAD
    return {
        "status": "PERFORM AI Backend running",
        "endpoints": {
            "pushup": "/pushup/analyze",
            "clean_jerk": "/weightlifting/clean-jerk/analyze",
            "upload_video": "/upload-video/",
            "processed_videos": "/local-videos/<filename>"
        }
    }

print("✅ Routers registered successfully")
=======
    print("  ℹ️   GET / — root endpoint called")
    return {
        "status": "PERFORM AI Backend running",
        "endpoints": {

            # Video analysis
            "pushup_video_analyze": "POST /pushup/analyze",
            "dip_analyze":          "POST /dips/analyze",
            "clean_jerk_analyze":   "POST /weightlifting/clean-jerk/analyze",
            "upload_video":         "POST /upload-video/",
            "processed_videos":     "GET  /local-videos/<filename>",

            # Live camera
            "pushup_live_start":  "POST /pushup/live/start?camera=0",
            "pushup_live_stop":   "POST /pushup/live/stop",
            "pushup_live_stats":  "GET  /pushup/live/stats",
            "pushup_live_stream": "GET  /pushup/live/stream   (MJPEG)",
            "pushup_live_health": "GET  /pushup/live/health",

            # Docs
            "swagger_ui": "GET  /docs",
            "redoc":      "GET  /redoc",
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
    print("    POST  http://localhost:8000/upload-video/")
    print("    GET   http://localhost:8000/local-videos/<filename>")
    print()
    print("  LIVE CAMERA (PUSHUP)")
    print("    POST  http://localhost:8000/pushup/live/start?camera=0")
    print("    POST  http://localhost:8000/pushup/live/stop")
    print("    GET   http://localhost:8000/pushup/live/stats")
    print("    GET   http://localhost:8000/pushup/live/stream")
    print("    GET   http://localhost:8000/pushup/live/health")
    print("=" * 60 + "\n")

@app.on_event("shutdown")
async def on_shutdown():
    print("\n  ⏹   PERFORM AI Backend shutting down...")

    try:
        from app.api.pushup_live_api import _session
        if _session and _session._running:
            print("  ⏹   Stopping active live session...")
            _session.stop()
            print("  ✅  Live session stopped cleanly")
    except Exception as e:
        print(f"  ⚠️   Could not stop live session on shutdown: {e}")

    print("  👋  Goodbye!\n")

print("\n✅  main.py loaded — all routers registered")

>>>>>>> ab5955a2fa4f135a5a143c0b766fb9db6514fa69
