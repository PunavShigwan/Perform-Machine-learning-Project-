import os
import shutil
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render

FASTAPI_ANALYZE_URL = "http://localhost:8000/pushup/analyze"


def home(request):
    return render(request, "frontend/index.html")


def exercises(request):
    return render(request, "frontend/exercises.html")


def pushup(request):
    return render(request, "frontend/pushup.html")


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
