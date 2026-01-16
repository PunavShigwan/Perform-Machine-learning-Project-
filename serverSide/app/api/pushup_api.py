from fastapi import APIRouter, UploadFile, File
import shutil
import os

print("📦 Loading pushup_api...")

try:
    from app.services.pushup.pushup_service import analyze_pushup_video
    print("✅ pushup_service imported")
except Exception as e:
    print("❌ ERROR importing pushup_service:", e)
    raise e

from app.schema.pushup_schema import PushupAnalysisResponse

router = APIRouter(prefix="/pushup", tags=["Pushup"])

UPLOAD_DIR = "app/uploads/input"
OUTPUT_DIR = "app/uploads/output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/analyze", response_model=PushupAnalysisResponse)
async def analyze_pushup(video: UploadFile = File(...)):
    print("📥 Video received:", video.filename)

    input_path = os.path.join(UPLOAD_DIR, video.filename)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{video.filename}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    print("🎥 Video saved to:", input_path)

    result = analyze_pushup_video(input_path, output_path)

    print("✅ Returning analysis response")
    return result
