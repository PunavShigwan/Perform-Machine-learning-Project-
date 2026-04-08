from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil, os, uuid, traceback, json
from pprint import pprint

print("📦 Loading dip_api...")

try:
    from app.services.dip.dip_service import analyze_dip_video
    print("✅ dip_service imported")
except Exception as e:
    print("❌ ERROR importing dip_service:", e)
    raise e

from app.schema.dip_schema import DipAnalysisResponse

router = APIRouter(tags=["Dips"])

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR  = os.path.join(BASE_DIR, "app", "uploads", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "app", "uploads", "output")

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📁 INPUT_DIR :", INPUT_DIR)
print("📁 OUTPUT_DIR:", OUTPUT_DIR)


@router.post("/analyze", response_model=DipAnalysisResponse)
async def analyze_dip(video: UploadFile = File(...)):
    try:
        print("\n================ DIP ANALYSIS START ================")
        print("📥 Video received:", video.filename)

        uid            = str(uuid.uuid4())
        safe_name      = f"{uid}_{video.filename}"
        input_path     = os.path.join(INPUT_DIR,  safe_name)
        processed_name = f"processed_{safe_name}"
        output_path    = os.path.join(OUTPUT_DIR, processed_name)

        with open(input_path, "wb") as buf:
            shutil.copyfileobj(video.file, buf)
        print("🎥 Input saved →", input_path)

        # ── Run analysis ──────────────────────────────────────────────────────
        result = analyze_dip_video(input_path, output_path)

        # ── Resolve final video filename (may change if ffmpeg absent) ────────
        saved_path     = result.pop("_saved_path", output_path)
        saved_filename = os.path.basename(saved_path)
        result["processed_video_url"] = (
            f"http://localhost:8000/local-videos/{saved_filename}"
        )

        print("\n📤 API RESPONSE:")
        pprint(result)
        try:
            json.dumps(result)
            print("✅ Valid JSON")
        except Exception as je:
            print("❌ JSON ERROR:", je)

        print(f"💾 Video saved → {saved_path}")
        print("================ DIP ANALYSIS END ================\n")

        return result

    except Exception as e:
        print("\n🔥 DIP API ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))