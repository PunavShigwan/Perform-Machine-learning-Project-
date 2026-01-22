import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

FASTAPI_ANALYZE_URL = "http://localhost:8000/analyze"


# ---------- PAGE RENDERING ----------
def home(request):
    return render(request, "frontend/index.html")


def exercises(request):
    return render(request, "frontend/exercises.html")


def pushup(request):
    return render(request, "frontend/pushup.html")


# ---------- AJAX UPLOAD → FASTAPI ----------
@csrf_exempt
def upload_video_ajax(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=400)

    video = request.FILES.get("video-blob") or request.FILES.get("video")
    if not video:
        return JsonResponse({"ok": False, "error": "No video received"}, status=400)

    try:
        response = requests.post(
            FASTAPI_ANALYZE_URL,
            files={"file": (video.name, video.read(), video.content_type)},
            timeout=300
        )

        return JsonResponse({
            "ok": True,
            "result": response.json()
        })

    except requests.exceptions.RequestException as e:
        return JsonResponse({
            "ok": False,
            "error": f"ML server error: {str(e)}"
        }, status=500)
