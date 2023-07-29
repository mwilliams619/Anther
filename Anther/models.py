from django.db import models

class Song(models.Model):
    # Song-specific information
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)
    danceability = models.FloatField()  # Use FloatField for decimal values
    energy = models.FloatField()       # Use FloatField for decimal values
    mode = models.IntegerField()
    valence = models.FloatField()      # Use FloatField for decimal values
    tempo = models.FloatField()        # Use FloatField for decimal values
    uri = models.TextField()
    key = models.IntegerField()
    popularity = models.IntegerField()
    genre = models.CharField(max_length=255)
    
    # Relationship with Artist model (ForeignKey: Many songs can have one artist)
    artist = models.ForeignKey('Artist', on_delete=models.CASCADE, related_name='songs')

    # Other fields for additional song-related information can be added here

    def __str__(self):
        return self.name


class SongRelationship(models.Model):
    id = models.BigAutoField(primary_key=True)
    # Foreign keys to represent the relationship between two songs
    song_a = models.ForeignKey('Song', related_name='related_songs_a', on_delete=models.CASCADE)
    song_b = models.ForeignKey('Song', related_name='related_songs_b', on_delete=models.CASCADE)

    # Additional fields for the relationship, if necessary
    similarity_score = models.FloatField()

    def __str__(self):
        return f"Song Relationship: {self.song_a.name} - {self.song_b.name}"


class Artist(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=100)
    uri = models.TextField()

    # Songs by the artist (One-to-Many relationship with Song model)
    tracks = models.ManyToManyField('Song', related_name='featured_artists')

    # Playlists featuring the artist (Many-to-Many relationship with Playlist model)
    playlists_featured_on = models.ManyToManyField('Playlist', related_name='featured_artists')

    # Other fields for additional artist-related information can be added here

    def __str__(self):
        return self.name


class Playlist(models.Model):
    id = models.BigAutoField(primary_key=True)
    # Playlist metadata information
    name = models.CharField(max_length=100)
    owner = models.JSONField()
    follow_count = models.IntegerField()
    description = models.TextField()
    cover_image = models.URLField()
    uri = models.TextField()

    # Songs on the playlist (Many-to-Many relationship with Song model)
    songs = models.ManyToManyField('Song', related_name='playlists')

    # Other fields for additional playlist-related information can be added here

    def __str__(self):
        return f"{self.get_display_name()} Playlist - {self.name}:\n {self.description[:50]}"

    def get_display_name(self):
        # Extract the display name from the owner dictionary
        return self.owner.get('display_name', 'Unknown')
    

class MailingList(models.Model):
    id = models.BigAutoField(primary_key=True)
    email = models.EmailField(unique=True)

    

