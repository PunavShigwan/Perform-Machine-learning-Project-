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
]
