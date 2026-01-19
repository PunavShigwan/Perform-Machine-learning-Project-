from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import traceback

print("📦 Loading pushup_api...")

try:
    from app.services.pushup.pushup_service import analyze_pushup_video
    print("✅ pushup_service imported")
except Exception as e:
    print("❌ ERROR importing pushup_service:", e)
    raise e

from app.schema.pushup_schema import PushupAnalysisResponse

# ❌ NO prefix here (prefix is applied in main.py)
router = APIRouter(tags=["Pushup"])

# Directories
INPUT_DIR = os.path.join("app", "uploads", "input")
OUTPUT_DIR = os.path.join("app", "uploads", "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/analyze", response_model=PushupAnalysisResponse)
async def analyze_pushup(video: UploadFile = File(...)):
    try:
        print("📥 Video received:", video.filename)

        safe_name = f"{uuid.uuid4()}_{video.filename}"
        input_path = os.path.join(INPUT_DIR, safe_name)
        output_path = os.path.join(OUTPUT_DIR, f"processed_{safe_name}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print("🎥 Video saved to:", input_path)

        result = analyze_pushup_video(input_path, output_path)

        print("✅ Returning analysis response")
        return result

    except Exception as e:
        print("🔥 PUSHUP API ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
