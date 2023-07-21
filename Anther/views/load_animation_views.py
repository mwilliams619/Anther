from django.http import JsonResponse
import os

def get_animation_frames(request):
    frames_directory = 'static/images/animation_frames'  # Update with the correct path to your frames folder
    frames = os.listdir(frames_directory)
    frames = [os.path.join('static/images/animation_frames', frame) for frame in frames]
    frames.sort()
    return JsonResponse(frames, safe=False)
