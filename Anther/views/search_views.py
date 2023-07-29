from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.db import connection
from django.shortcuts import render
from Anther.Services.spotifyClassDef import PlaylistClass, ArtistClass, TrackClass, CrawlClass

db = connection.cursor()

# Assuming you have created a template called 'search.html' with the search bar and dropdown menu as described earlier

@require_GET
def search_results(request):
    # Get the search query and search category from the request
    search_query = request.GET.get('query', '')
    search_category = request.GET.get('search_category', 'track')  # Default to track if not provided

    # Create a SpotSuper instance based on the selected search category
    if search_category == 'track':
        spot_instance = TrackClass(name=search_query)  # Pass your Django db connection here
        # song_details = spot_instance.song_properties()
        search_results = spot_instance.basic_list_search()
        # print(song_details)
        print(search_results)
        # search_results = [{'name': song_details['name'],'artist': song_details['artist']}]

    elif search_category == 'artist':
        spot_instance = ArtistClass(name=search_query)  # Pass your Django db connection here
        # search_results = spot_instance.multi_song_properties()
        search_results = spot_instance.basic_list_search()
        print(search_results)

    elif search_category == 'playlist':
        spot_instance = PlaylistClass(name=search_query)  # Pass your Django db connection here
        # search_results = spot_instance.multi_song_properties()
        search_results = spot_instance.basic_list_search()
        print(search_results)
    
    elif search_category == 'thicc':
        search_results = crawl(search_query)
    
    elif search_category == 'contactable-playlist':
        spot_instance = PlaylistClass(name=search_query)
        search_results = spot_instance.contactable_search()

    else:
        return JsonResponse({'error': 'Invalid search category.'}, status=400)

    # Call the search method of SpotSuper and get the search results
    

    # Return the search results as JSON data
    return JsonResponse(search_results, safe=False)

def crawl(query):
    search_query = query
    spotinstance = CrawlClass(name=search_query)
    spotinstance.crawling()
    return "You finished loading the DB"

def search_view(request):
    return render(request, 'search.html')

def link_find(request):
    playlist_name = request.GET.get('param_name') #Get the playlist id in this line so you can 
    spot_instance = PlaylistClass(name=playlist_name)
    links = spot_instance.similarness()
    response_data = []

    # Iterate through the link_list
    for link in links:
        # Get the song names of both songs and similarity score
        song_a_name = link.song_a.name
        song_b_name = link.song_b.name
        similarity_score = link.similarity_score

        # Create a dictionary to store the data
        data_dict = {
            'song_a_name': song_a_name,
            'song_b_name': song_b_name,
            'similarity_score': similarity_score,
        }

        # Append the dictionary to the response_data list
        response_data.append(data_dict)
    print(response_data)
    return JsonResponse(response_data, safe=False)
