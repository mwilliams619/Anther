from django.urls import path
import Anther.views.playlist_views as views

urlpatterns = [
    path('playlists/', views.playlist_view, name='playlist-view'),
    path('get_playlist_data/', views.get_playlist_data, name='get_playlist_data'),
     path('get_tracklist/', views.get_tracklist, name='get_tracklist'),
]