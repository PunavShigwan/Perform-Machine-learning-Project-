from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import traceback
import json
from pprint import pprint

print("📦 Loading pushup_api...")

try:
    from app.services.pushup.pushup_service import analyze_pushup_video
    print("✅ pushup_service imported")
except Exception as e:
    print("❌ ERROR importing pushup_service:", e)
    raise e

from app.schema.pushup_schema import PushupAnalysisResponse

router = APIRouter(tags=["Pushup"])

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "app", "uploads", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "app", "uploads", "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📁 INPUT_DIR:", INPUT_DIR)
print("📁 OUTPUT_DIR:", OUTPUT_DIR)

@router.post("/analyze", response_model=PushupAnalysisResponse)
async def analyze_pushup(video: UploadFile = File(...)):
    try:
        print("\n================ PUSHUP ANALYSIS START ================")
        print("📥 Video received:", video.filename)

        uid = str(uuid.uuid4())
        safe_name = f"{uid}_{video.filename}"

        input_path = os.path.join(INPUT_DIR, safe_name)
        processed_name = f"processed_{safe_name}"
        output_path = os.path.join(OUTPUT_DIR, processed_name)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print("🎥 Video saved to:", input_path)

        result = analyze_pushup_video(input_path, output_path)

        processed_url = f"http://localhost:8000/local-videos/{processed_name}"
        result["processed_video_url"] = processed_url

        print("\n📤 API RESPONSE (DICT):")
        pprint(result)

        json.dumps(result)  # validate JSON

        print("================ PUSHUP ANALYSIS END ================\n")
        return result

    except Exception as e:
        print("\n🔥 PUSHUP API ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))