from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
import traceback
import json
from pprint import pprint

print("📦 Loading clean_jerk_api...")

try:
    from app.services.weightlifting.clean_jerk.clean_jerk_service import analyze_clean_jerk_video
    print("✅ clean_jerk_service imported")
except Exception as e:
    print("❌ ERROR importing clean_jerk_service:", e)
    raise e

from app.schema.clean_jerk_schema import CleanJerkAnalysisResponse

router = APIRouter(tags=["Clean & Jerk"])

# =====================================================
# DIRECTORIES
# =====================================================
INPUT_DIR = os.path.join("app", "uploads", "input")
OUTPUT_DIR = os.path.join("app", "uploads", "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# API
# =====================================================
@router.post("/clean-jerk/analyze", response_model=CleanJerkAnalysisResponse)
async def analyze_clean_jerk(video: UploadFile = File(...)):
    try:
        print("\n================ CLEAN & JERK ANALYSIS START ================")
        print("📥 Video received:", video.filename)

        safe_name = f"{uuid.uuid4()}_{video.filename}"
        input_path = os.path.join(INPUT_DIR, safe_name)
        output_path = os.path.join(OUTPUT_DIR, f"processed_{safe_name}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print("🎥 Video saved to:", input_path)

        # ===============================
        # ANALYSIS
        # ===============================
        result = analyze_clean_jerk_video(input_path, output_path)

        # ===============================
        # 🔎 DEBUG LOGS
        # ===============================
        print("\n📤 API RESPONSE (DICT):")
        pprint(result)

        try:
            json.dumps(result)
            print("✅ Response is valid JSON")
        except Exception as json_err:
            print("❌ JSON SERIALIZATION ERROR:", json_err)

        print("================ CLEAN & JERK ANALYSIS END ================\n")

        return result

    except Exception as e:
        print("\n🔥 CLEAN & JERK API ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))