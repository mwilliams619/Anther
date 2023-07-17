from django.urls import path
import views.playlist_views as views

urlpatterns = [
    path('playlists/', views.playlist_view, name='playlist-view'),
]