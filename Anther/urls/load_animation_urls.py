from django.urls import path
import Anther.views.load_animation_views as views

urlpatterns = [
    path('loading/', views.get_animation_frames, name='get_animation_frames'),
    path('loading_diagram/', views.get_diagram_animation_frames, name='get_diagram_frames'),
    # Add other URL patterns if needed
]