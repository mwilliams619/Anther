from django.views.generic import ListView
from django.shortcuts import render
# from models import Playlist  


def playlist_view(request):
    return render(request, 'playlist_view.html')
