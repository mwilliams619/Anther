from Anther.Services.spotifyClassDef import Playlist, Artist, Track, SpotSuper
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.db import connection
from django.shortcuts import render

db = connection.cursor()

# Assuming you have created a template called 'search.html' with the search bar and dropdown menu as described earlier

@require_GET
def search_results(request):
    # Get the search query and search category from the request
    search_query = request.GET.get('query', '')
    search_category = request.GET.get('search_category', 'track')  # Default to track if not provided

    # Create a SpotSuper instance based on the selected search category
    if search_category == 'track':
        spot_instance = SpotSuper(name=search_query, category='track', connection=None)  # Pass your Django db connection here
        song_details = spot_instance.song_properties()
        print(song_details)
        search_results = [{'name': song_details['name'],'artist': song_details['artist']}]

    elif search_category == 'artist':
        spot_instance = SpotSuper(name=search_query, category='artist', connection=None)  # Pass your Django db connection here
        search_results = spot_instance.multi_song_properties()

    elif search_category == 'playlist':
        spot_instance = SpotSuper(name=search_query, category='playlist', connection=None)  # Pass your Django db connection here
        search_results = spot_instance.multi_song_properties()
    else:
        return JsonResponse({'error': 'Invalid search category.'}, status=400)

    # Call the search method of SpotSuper and get the search results
    

    # Return the search results as JSON data
    return JsonResponse(search_results, safe=False)

def search_view(request):
    return render(request, 'search.html')
