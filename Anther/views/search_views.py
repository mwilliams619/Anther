from Anther.Services.spotifyClassDef import Playlist, Artist, Track, SpotSuper
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.db import connection
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
    elif search_category == 'artist':
        spot_instance = SpotSuper(name=search_query, category='artist', connection=None)  # Pass your Django db connection here
    elif search_category == 'playlist':
        spot_instance = SpotSuper(name=search_query, category='playlist', connection=None)  # Pass your Django db connection here
    else:
        return JsonResponse({'error': 'Invalid search category.'}, status=400)

    # Call the search method of SpotSuper and get the search results
    search_results = spot_instance.search()

    # Return the search results as JSON data
    return JsonResponse(search_results)
