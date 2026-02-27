from django.urls import path
from . import views

app_name = 'frontend'

urlpatterns = [
    path('', views.home, name='home'),                # http://127.0.0.1:8000/
    path('exercises/', views.exercises, name='exercises'),  # /exercises/
    path('pushup/', views.pushup, name='pushup'),     # /pushup/
    path('clean_and_jerk/', views.clean_and_jerk, name='clean_and_jerk'),     # /clean_and_jerk/
    path('upload_video_ajax/', views.upload_video_ajax, name='upload_video_ajax'),
    path("upload-cleanjerk/", views.upload_cleanjerk_ajax, name="upload_cleanjerk"),
]
