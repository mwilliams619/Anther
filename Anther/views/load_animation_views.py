from django.contrib.staticfiles.storage import staticfiles_storage
from django.http import JsonResponse
import os

def get_animation_frames(request):
       frames_directory = 'images/animation_frames'  # Relative path within static files
       storage = staticfiles_storage
       frames = storage.listdir(frames_directory)[1]  # [1] to get files, not directories
       frames = [storage.url(f'{frames_directory}/{frame}') for frame in frames]
       frames.sort()
       return JsonResponse(frames, safe=False)

def get_diagram_animation_frames(request):
    frames_directory = 'images/planning_animation'  # Relative path within static files
    storage = staticfiles_storage
    frames = storage.listdir(frames_directory)[1]  # [1] to get files, not directories
    frames = [storage.url(f'{frames_directory}/{frame}') for frame in frames]
    frames.sort()
    return JsonResponse(frames, safe=False)