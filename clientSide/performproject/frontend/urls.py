from django.urls import path
from . import views

app_name = 'frontend'

urlpatterns = [
    path('', views.home, name='home'),                # http://127.0.0.1:8000/
    path('exercises/', views.exercises, name='exercises'),  # /exercises/
    path('pushup/', views.pushup, name='pushup'),     # /pushup/
    path('clean_and_jerk/', views.clean_and_jerk, name='clean_and_jerk'),     # /clean_and_jerk/
    path('dips/', views.dips, name='dips'),
    path('upload_video_ajax/', views.upload_video_ajax, name='upload_video_ajax'),
    path("upload-cleanjerk/", views.upload_cleanjerk_ajax, name="upload_cleanjerk"),
    path("upload-dips/", views.upload_dip_ajax, name="upload_dip_ajax"),

    # Live pushup (proxy to FastAPI)
    path("pushup/live/start/",  views.live_start,        name="live_start"),
    path("pushup/live/stop/",   views.live_stop,         name="live_stop"),
    path("pushup/live/stats/",  views.live_stats,        name="live_stats"),
    path("pushup/live/stream/", views.live_stream_proxy, name="live_stream"),

    # Squat
    path('squat/', views.squat, name='squat'),
    path('upload-squat/', views.upload_squat_ajax, name='upload_squat_ajax'),
    path('squat/live/start/', views.squat_live_start, name='squat_live_start'),
    path('squat/live/stop/', views.squat_live_stop, name='squat_live_stop'),
    path('squat/live/stats/', views.squat_live_stats, name='squat_live_stats'),
    path('squat/live/stream/', views.squat_live_stream, name='squat_live_stream'),

    # Clean & Jerk Live
    path('cleanjerk/live/start/', views.cj_live_start, name='cj_live_start'),
    path('cleanjerk/live/stop/', views.cj_live_stop, name='cj_live_stop'),
    path('cleanjerk/live/stats/', views.cj_live_stats, name='cj_live_stats'),
    path('cleanjerk/live/stream/', views.cj_live_stream, name='cj_live_stream'),
    path('cleanjerk/live/reset/', views.cj_live_reset, name='cj_live_reset'),
]
