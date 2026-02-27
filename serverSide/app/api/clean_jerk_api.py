"""
app/api/clean_jerk_api.py
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil, os, uuid, traceback, json, logging
from pprint import pformat

logger = logging.getLogger("clean_jerk_api")
print("📦 Loading clean_jerk_api...")

try:
    from app.services.weightlifting.clean_jerk.clean_jerk_service import analyze_clean_jerk_video
    print("✅ clean_jerk_service imported")
except Exception as e:
    print("❌ ERROR importing clean_jerk_service:", e)
    traceback.print_exc()
    raise e

from app.schema.clean_jerk_schema import CleanJerkAnalysisResponse

router = APIRouter(tags=["Clean & Jerk"])

INPUT_DIR  = os.path.join("app", "uploads", "input")
OUTPUT_DIR = os.path.join("app", "uploads", "output")
os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT    = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_SIZE_BYTES = 200 * 1024 * 1024


def _cleanup(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"[cleanup] {path}: {e}")


@router.post("/clean-jerk/analyze", response_model=CleanJerkAnalysisResponse)
async def analyze_clean_jerk(video: UploadFile = File(...)):
    print("\n================ CLEAN & JERK ANALYSIS START ================")
    print(f"📥 Received: {video.filename}  type={video.content_type}")

    # ── Validate ──────────────────────────────────────────────────────────────
    if not video.filename:
        raise HTTPException(400, "No filename provided.")
    ext = os.path.splitext(video.filename)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {', '.join(ALLOWED_EXT)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    safe_name   = f"{uuid.uuid4()}_{video.filename}"
    input_path  = os.path.join(INPUT_DIR, safe_name)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{safe_name.rsplit('.', 1)[0]}.mp4")

    try:
        with open(input_path, "wb") as buf:
            shutil.copyfileobj(video.file, buf)
        size_kb = os.path.getsize(input_path) // 1024
        print(f"💾 Saved: {input_path} ({size_kb} KB)")
        if size_kb == 0:
            raise ValueError("Uploaded file is empty.")
        if os.path.getsize(input_path) > MAX_SIZE_BYTES:
            raise ValueError(f"File too large ({size_kb//1024} MB). Max 200 MB.")
    except (ValueError, HTTPException) as e:
        _cleanup(input_path)
        raise HTTPException(400, str(e))
    except Exception as e:
        _cleanup(input_path)
        print(f"❌ Save failed: {e}"); traceback.print_exc()
        raise HTTPException(500, f"Failed to save file: {e}")

    # ── Analyse ───────────────────────────────────────────────────────────────
    result = None
    try:
        result = analyze_clean_jerk_video(input_path, output_path)
    except Exception as e:
        print(f"\n🔥 UNCAUGHT exception from analyzer:")
        traceback.print_exc()
        _cleanup(input_path)
        raise HTTPException(500, detail={
            "error": True, "context": "analyzer_uncaught",
            "message": str(e), "traceback": traceback.format_exc(),
        })
    finally:
        _cleanup(input_path)

    # ── Structured error from service ────────────────────────────────────────
    if isinstance(result, dict) and result.get("error"):
        print(f"\n🔥 ANALYZER ERROR:")
        print(f"   context  : {result.get('context')}")
        print(f"   message  : {result.get('message')}")
        print(f"   traceback:\n{result.get('traceback', '')}")
        raise HTTPException(500, detail={
            "error":     True,
            "context":   result.get("context", "unknown"),
            "message":   result.get("message", "Unknown error"),
            "traceback": result.get("traceback", ""),
        })

    # ── Pretty print result ───────────────────────────────────────────────────
    pt = result.get("phase_timing", {})
    print("\n📤 RESULT SUMMARY:")
    print(f"   machine_call          : {result.get('machine_call')}")
    print(f"   machine_reason        : {result.get('machine_reason')}")
    print(f"   form_accuracy_percent : {result.get('form_accuracy_percent')}%")
    print(f"   form_frozen           : {result.get('form_frozen')}")
    print(f"   total_frames          : {result.get('total_frames')}")
    print(f"   active_lift_frames    : {result.get('active_lift_frames')}")
    print(f"   bad_frames            : {result.get('bad_frames')}")
    print(f"   lift_duration         : {pt.get('total_lift_duration_seconds')}s")
    print(f"   fastest_phase         : {pt.get('fastest_phase')}")
    print(f"   slowest_phase         : {pt.get('slowest_phase')}")
    for st, s in result.get("stage_summary", {}).items():
        print(f"   [{st}] detected={s.get('detected')}  "
              f"dur={s.get('duration_seconds')}s  "
              f"form_ok={s.get('form_ok')}  "
              f"issues={s.get('top_issues')}")

    # ── Serialisation guard ───────────────────────────────────────────────────
    try:
        json.dumps(result)
        print("✅ JSON valid")
    except Exception as e:
        print(f"❌ JSON ERROR: {e}")
        raise HTTPException(500, f"Serialisation failed: {e}")

    print("================ CLEAN & JERK ANALYSIS END ================\n")
    return JSONResponse(content=result)