from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

print("🚀 Starting FastAPI application...")

# =====================================================
# CREATE APP
# =====================================================
app = FastAPI(title="PERFORM AI Backend")

# =====================================================
# CORS (DEV MODE)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # DEV ONLY
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
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

# =====================================================
# VIDEO UPLOAD + PROCESS ENDPOINT
# =====================================================
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
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
    }

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
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