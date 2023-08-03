from django.views.generic import ListView
from django.shortcuts import render
from Anther.models import Artist, Playlist, Song, SongRelationship
from django.http import JsonResponse
import random
from Anther.Services.spotifyClassDef import PlaylistClass, ArtistClass, TrackClass, CrawlClass


def playlist_view(request):
    return render(request, 'playlist_view.html')

def get_playlist_data(request):
    playlists = Playlist.objects.all()  # Query your playlists
    # I'm going to make the sizes random for now the value should correspond to the affinity to the playlist though
    data = [{'id': playlist.id, 'name': playlist.name, 'value': random.randint(15, 80)} for playlist in playlists[37:65]]
    return JsonResponse(data, safe=False)

def get_tracklist(request):

    playlist_name = request.GET.get('name')
    playlist_instance = PlaylistClass(name=playlist_name)

    graph_data = playlist_instance.similarness()
    return JsonResponse(graph_data, safe=False)