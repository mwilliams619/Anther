from django.db import models

class Playlist(models.Model):
    name = models.CharField(max_length=80)
    songs = models.CharField