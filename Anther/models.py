from django.db import models

class SongProps(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)
    artist = models.CharField(max_length=255, unique=True, default=None)
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
    
class Artist(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=100)

    # Songs by the artist (One-to-Many relationship with Song model)
    tracks = models.ManyToManyField('Song', related_name='featured_artists')

    # Playlists featuring the artist (Many-to-Many relationship with Playlist model)
    playlists_featured_on = models.ManyToManyField('Playlist', related_name='featured_artists')

    # Other fields for additional artist-related information can be added here

    def __str__(self):
        return self.name

class Song(models.Model):
    # Song-specific information
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
    
    # Relationship with Artist model (ForeignKey: Many songs can have one artist)
    artist = models.ForeignKey('Artist', on_delete=models.CASCADE, related_name='songs')

    # Other fields for additional song-related information can be added here

    def __str__(self):
        return self.title

class Playlist(models.Model):
    id = models.BigAutoField(primary_key=True)
    # Playlist metadata information
    name = models.CharField(max_length=100)
    owner = models.CharField(max_length=100)
    number_of_likes = models.IntegerField()
    description = models.TextField()

    # Songs on the playlist (Many-to-Many relationship with Song model)
    songs = models.ManyToManyField('Song', related_name='playlists')

    # Other fields for additional playlist-related information can be added here

    def __str__(self):
        return f"{self.owner}'s Playlist - {self.name}:\n {self.description[:50]}"  # Customize the display of the playlist in the admin panel
    

