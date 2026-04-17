import os
import shutil
import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render

FASTAPI_BASE = "http://localhost:8000"
FASTAPI_ANALYZE_URL = f"{FASTAPI_BASE}/pushup/analyze"
FASTAPI_CLEAN_JERK_URL = f"{FASTAPI_BASE}/weightlifting/clean-jerk/analyze"
FASTAPI_DIP_URL = f"{FASTAPI_BASE}/dips/analyze"


def home(request):
    return render(request, "frontend/index.html")


def exercises(request):
    return render(request, "frontend/exercises.html")


def pushup(request):
    return render(request, "frontend/pushup.html")

def clean_and_jerk(request):
    return render(request, "frontend/clean_and_jerk.html")

def dips(request):
    return render(request, "frontend/dips.html")

@csrf_exempt
def upload_video_ajax(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)

    video = request.FILES.get("video-blob") or request.FILES.get("video")
    if not video:
        return JsonResponse({"ok": False, "error": "No video received"}, status=400)

    try:
        # 🔁 Send video to FastAPI
        response = requests.post(
            FASTAPI_ANALYZE_URL,
            files={"video": (video.name, video.read(), video.content_type)},
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        # 🎯 Get processed video path from FastAPI
        processed_path = result.get("output_video_path")
        if not processed_path:
            return JsonResponse(
                {"ok": False, "error": "FastAPI did not return output path"},
                status=500
            )

        # 🧹 Normalize Windows path
        processed_path = os.path.normpath(processed_path)

        if not os.path.exists(processed_path):
            return JsonResponse(
                {"ok": False, "error": f"Processed video not found: {processed_path}"},
                status=500
            )

        # 📂 Ensure Django media directory exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        filename = os.path.basename(processed_path)
        dest_path = os.path.join(settings.MEDIA_ROOT, filename)

        # 📄 Copy file from FastAPI → Django
        shutil.copy(processed_path, dest_path)

        return JsonResponse({
            "ok": True,
            "processed_video_url": settings.MEDIA_URL + filename,
            "result": result
        })

    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": str(e)},
            status=500
        )
@csrf_exempt
def upload_cleanjerk_ajax(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)

    video = request.FILES.get("video-blob") or request.FILES.get("video")

    if not video:
        return JsonResponse({"ok": False, "error": "No video received"}, status=400)

    try:
        response = requests.post(
            FASTAPI_CLEAN_JERK_URL,
            files={"video": (video.name, video.read(), video.content_type)},
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        processed_path = result.get("processed_video_path")

        if not processed_path:
            return JsonResponse(
                {"ok": False, "error": "FastAPI did not return output path"},
                status=500
            )

        processed_path = os.path.normpath(processed_path)

        if not os.path.exists(processed_path):
            return JsonResponse(
                {"ok": False, "error": f"Video not found: {processed_path}"},
                status=500
            )

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        filename = os.path.basename(processed_path)
        dest_path = os.path.join(settings.MEDIA_ROOT, filename)

        shutil.copy(processed_path, dest_path)

        return JsonResponse({
            "ok": True,
            "processed_video_url": settings.MEDIA_URL + filename,
            "result": result
        })

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

# =====================================================
# DIPS  —  Proxy to FastAPI /dips/analyze
# =====================================================

@csrf_exempt
def upload_dip_ajax(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)

    video = request.FILES.get("video-blob") or request.FILES.get("video")
    if not video:
        return JsonResponse({"ok": False, "error": "No video received"}, status=400)

    try:
        response = requests.post(
            FASTAPI_DIP_URL,
            files={"video": (video.name, video.read(), video.content_type)},
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        processed_url = result.get("processed_video_url", "")

        # The FastAPI response gives a full URL like http://localhost:8000/local-videos/...
        # We need to download the processed video and serve it from Django's media dir
        if processed_url:
            import urllib.request
            filename = os.path.basename(processed_url.split("?")[0])
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            dest_path = os.path.join(settings.MEDIA_ROOT, filename)

            # Try to copy from the FastAPI output dir first (faster than HTTP download)
            fastapi_output = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "..", "serverSide", "app", "uploads", "output", filename
                )
            )
            if os.path.exists(fastapi_output):
                shutil.copy(fastapi_output, dest_path)
            else:
                urllib.request.urlretrieve(processed_url, dest_path)

            return JsonResponse({
                "ok": True,
                "processed_video_url": settings.MEDIA_URL + filename,
                "result": result
            })

        return JsonResponse({"ok": True, "result": result})

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


# =====================================================
# LIVE PUSHUP  —  Proxy to FastAPI live endpoints
# =====================================================

@csrf_exempt
def live_start(request):
    """POST — start a live webcam session on the FastAPI server."""
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)
    camera = request.GET.get("camera", "0")
    try:
        r = requests.post(f"{FASTAPI_BASE}/pushup/live/start", params={"camera": camera}, timeout=10)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=502)


@csrf_exempt
def live_stop(request):
    """POST — stop the running live session and return final stats."""
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)
    try:
        r = requests.post(f"{FASTAPI_BASE}/pushup/live/stop", timeout=10)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=502)


def live_stats(request):
    """GET — poll live stats from the FastAPI session."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/pushup/live/stats", timeout=5)
        return JsonResponse(r.json(), status=r.status_code)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=502)


def live_stream_proxy(request):
    """GET — proxy the MJPEG stream so the browser can use a same-origin URL."""
    try:
        r = requests.get(f"{FASTAPI_BASE}/pushup/live/stream", stream=True, timeout=5)
        if r.status_code != 200:
            return JsonResponse({"ok": False, "error": "Stream not available"}, status=r.status_code)
        response = StreamingHttpResponse(
            r.iter_content(chunk_size=4096),
            content_type=r.headers.get("content-type", "multipart/x-mixed-replace; boundary=frame"),
        )
        return response
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=502)