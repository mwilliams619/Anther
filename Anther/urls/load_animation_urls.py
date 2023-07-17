from django.urls import path
import views.load_animation_views as views

urlpatterns = [
    path('loading/', views.get_animation_frames, name='get_animation_frames'),
    # Add other URL patterns if needed
]