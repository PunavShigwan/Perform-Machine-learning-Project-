"""
app/api/squat_api.py
FastAPI router for squat video upload & analysis.

Endpoint (prefix /squat applied in main.py):
  POST  /squat/analyze        Upload a video and get full squat analysis
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import traceback
import json
from pprint import pprint

print("📦 Loading squat_api...")

try:
    from app.services.squat.squat_service import analyze_squat_video
    print("✅ squat_service imported")
except Exception as e:
    print("❌ ERROR importing squat_service:", e)
    raise e

from app.schema.squat_schema import SquatAnalysisResponse

router = APIRouter(tags=["Squat"])

# ─────────────────────────────────────────────
#  DIRECTORIES
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# BASE_DIR → serverSide/

INPUT_DIR  = os.path.join(BASE_DIR, "app", "uploads", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "app", "uploads", "output")

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📁 INPUT_DIR :", INPUT_DIR)
print("📁 OUTPUT_DIR:", OUTPUT_DIR)

# ─────────────────────────────────────────────
#  ENDPOINT
# ─────────────────────────────────────────────
@router.post("/analyze", response_model=SquatAnalysisResponse)
async def analyze_squat(video: UploadFile = File(...)):
    """
    Upload an MP4 / MOV / AVI squat video.
    Returns rep counts, form score, per-rep fault log, and a URL
    to the annotated output video.
    """
    try:
        print("\n================ SQUAT ANALYSIS START ================")
        print("📥 Video received:", video.filename)

        uid        = str(uuid.uuid4())
        safe_name  = f"{uid}_{video.filename}"

        input_path       = os.path.join(INPUT_DIR,  safe_name)
        processed_name   = f"processed_{safe_name}"
        output_path      = os.path.join(OUTPUT_DIR, processed_name)

        # ── Save uploaded file ────────────────────────────────
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        print("🎥 Video saved to:", input_path)

        # ── Run ML pipeline ───────────────────────────────────
        result = analyze_squat_video(input_path, output_path)

        # ── Attach public video URL ───────────────────────────
        processed_url                 = f"http://localhost:8000/local-videos/{processed_name}"
        result["processed_video_url"] = processed_url

        # ── Log response ──────────────────────────────────────
        print("\n📤 API RESPONSE (DICT):")
        pprint(result)

        try:
            json.dumps(result)
            print("✅ Response is valid JSON")
        except Exception as json_err:
            print("❌ JSON SERIALIZATION ERROR:", json_err)

        print("================ SQUAT ANALYSIS END ================\n")
        return result

    except Exception as e:
        print("\n🔥 SQUAT API ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))