from django.views.generic import ListView
from models import Playlist  


class MyLisView(ListView):
    model = Playlist
