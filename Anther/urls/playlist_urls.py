from django.urls import path
import Anther.views.playlist_views as views

urlpatterns = [
    path('playlists/', views.playlist_view, name='playlist-view'),
]