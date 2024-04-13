from django.db import models
from django.conf import settings
from time import timezone

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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):
        return self.name
    
class ArtistFollower(models.Model):
    id = models.AutoField(primary_key=True)
    artist = models.ForeignKey('Artist', on_delete=models.CASCADE)  
    spotify_followers = models.IntegerField() # from API
    spotify_monthly_listeners = models.IntegerField() #user input
    ig_followers = models.IntegerField() # from API
    tiktok_followers = models.IntegerField()
    youtube_subscribers = models.IntegerField()
    date = models.DateField()

    def __str__(self):
        return f"{self.artist.name} had {self.followers} followers on {self.date}"


class Playlist(models.Model):
    id = models.BigAutoField(primary_key=True)
    # Playlist metadata information
    name = models.CharField(max_length=100)
    owner = models.JSONField()
    follow_count = models.IntegerField()
    description = models.TextField()
    cover_image = models.URLField()
    uri = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

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


class MoneySpent(models.Model):
    id = models.AutoField(primary_key=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    spent_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )

    def __str__(self):
        return f"{self.amount} on {self.category}"
    

class MoneyReceived(models.Model):
    id = models.AutoField(primary_key=True)
    PAYMENT_METHODS = [
        ('CA', 'Cash'),
        ('BT', 'Bank Transfer'),
        ('PP', 'PayPal'),
        ('CC', 'Credit Card'),
    ]
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    source = models.CharField(max_length=100) 
    description = models.TextField(blank=True)
    received_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    receipt_image = models.ImageField(upload_to='receipts', blank=True)
    payment_method = models.CharField(
        max_length=2,
        choices=PAYMENT_METHODS,
        default='CA'
    )
    
    def __str__(self):
        return f"{self.amount} from {self.source}"
    

class SongMetrics(models.Model):
    id = models.AutoField(primary_key=True)
    song_title = models.CharField(max_length=100)
    artist = models.CharField(max_length=100)

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    song = models.OneToOneField(Song, on_delete=models.CASCADE)
    playlists = models.ManyToManyField(Playlist)

    spotify_uri = models.CharField(max_length=200, blank=True)
    spotify_streams = models.IntegerField(default=0)
    apple_music_streams = models.IntegerField(null=True, blank=True)

    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.song_title} by {self.artist}"

class Campaign(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    start_date = models.DateField()
    end_date = models.DateField()

    merch = models.ForeignKey('Merch', on_delete=models.CASCADE)
    
    platforms = models.ManyToManyField('Platform', related_name='campaigns')

    budget = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)

    money_spent = models.OneToOneField('MoneySpent', on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.name

class Platform(models.Model):
    PLATFORM_CHOICES = [
        ('FB', 'Facebook'),
        ('IG', 'Instagram'),
        ('TW', 'Twitter'),
        ('SP', 'Spotify'),
        ('YT', 'YouTube'),
        ('LV', 'Real World'),
    ]
    name = models.CharField(max_length=2, choices=PLATFORM_CHOICES)

class Merch(models.Model):
    CATEGORY_CHOICES = [
        ('TS', 'T-Shirts'),
        ('HT', 'Hats'),
        ('HD', 'Hoodie'),
        ('PT', 'Posters'),
        ('OT', 'Other'),
    ]

    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=2, choices=CATEGORY_CHOICES)
    description = models.TextField(blank=True)

    sale_price = models.DecimalField(max_digits=6, decimal_places=2)
    cost = models.DecimalField(max_digits=6, decimal_places=2)
    profit = models.DecimalField(max_digits=6, decimal_places=2, editable=False)
    
    inventory = models.IntegerField(default=0)
    sold = models.IntegerField(default=0)

    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        self.profit = self.sale_price - self.cost
        super().save(*args, **kwargs) 

    def __str__(self):
        return self.name