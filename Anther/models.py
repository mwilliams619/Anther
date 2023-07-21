from django.db import models

class SongProps(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)
    danceability = models.IntegerField()
    energy = models.IntegerField()
    mode = models.IntegerField()
    valence = models.IntegerField()
    tempo = models.IntegerField()
    uri = models.TextField()
    key = models.IntegerField()
    popularity = models.IntegerField()
    genre = models.CharField(max_length=255)

    def __str__(self):
        return self.name
