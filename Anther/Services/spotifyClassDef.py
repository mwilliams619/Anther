import copy
import sqlite3
from json import dump
from math import log10
from statistics import mean

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import spotipy
from django.db import connection
from django.db import transaction
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import \
    SpotifyClientCredentials  # To access authorised Spotify data

from Anther.models import Artist, Playlist, Song, SongRelationship

# Import Login credentials from .env file

#  visit spotify api website to create an app and receive a client id and client secret
client_id = '6825dd9b8f2740eaa12044b0681a3a78'
client_secret = 'f8822161b11b46f6b6e624b015aaa4d7'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API

# db = connection.cursor()

# db.execute("CREATE TABLE IF NOT EXISTS song_props (name TEXT PRIMARY KEY NOT NULL, danceability INTEGER NOT NULL,"
#            "energy INTEGER NOT NULL, mode INTEGER NOT NULL, valence INTEGER NOT NULL, tempo INTEGER NOT NULL,"
#            "uri TEXT NOT NULL, key INTEGER NOT NULL, popularity INTEGER NOT NULL, genre TEXT NOT NULL)")


class SpotSuper:

    def __init__(self, name, category, model):
        self.name = name
        self._category = category
        self.model = model
        self._uri = self.search()
        # self.connection_cursor = connection
        # self.cursor = self.connection_cursor
        if self._category == "track":
            self.catalog = None
        if self.model == "thicc":
            self.catalog = None
        else:
            self.catalog = self.track_list()

    def basic_list_search(self, limit=1): 
        all_results = sp.search(q=self._category + ':' + self.name, type=self._category, limit=limit)
        result_list = []
        if all_results and self._category + 's' in all_results and 'items' in all_results[self._category + 's']:
            if self.model == 'track':
                for item in all_results[self._category + 's']['items']:
                    props = self.song_properties(track = (item['name'], item['uri']))
                    artprop = props['artist']
                    if isinstance(artprop, str):     
                        thisartist = ArtistClass(name=artprop)
                    elif isinstance(artprop.name, str): 
                        thisartist = ArtistClass(name=artprop.name)
                    artobj, _ =  Artist.objects.get_or_create(name= thisartist.name, uri=thisartist._uri)
                    songobj = Song.objects.get(name=props['name'])
                    artobj.tracks.add(songobj)

                    result_list.append((item['name'], item['artists'][0]['name'], item['uri']))

            elif self.model == 'artist':
                for item in all_results[self._category + 's']['items']:
                    try:
                        artist = Artist.objects.get(name=item['name'])
                    except Artist.DoesNotExist:
                        artist = Artist.objects.create(name=item['name'], uri=item['uri'])
                    thisinstance = ArtistClass(name=item['name'])
                    tracks = thisinstance.track_list()
                    # for track in tracks:
                    #     song = TrackClass(name=track).song_properties()
                    #     songobj = Song.objects.get(name=track)
                    #     artist.tracks.add(songobj)

                    result_list.append((item['name'], item['genres']))
                # Assuming you have a field named 'uri' in the Artist model
            elif self.model == 'playlist':
                for item in all_results[self._category + 's']['items']:
                    thisinstance = PlaylistClass(name=item['name'])
                    result_list.append((item['name'], item['owner']['display_name'], item['uri']))

        return result_list



    def search(self):
        """Search for either track/playlist/artist item and return the URI."""
        try:
            if self.model == 'track':
                song_props = Song.objects.get(name=self.name)
                return song_props.uri
            elif self.model == 'artist':
                artist = Artist.objects.get(name=self.name)
                # Assuming you have a field named 'uri' in the Artist model
                return artist.uri
            elif self.model in ['playlist', 'thicc']:
                try:
                    playlist = Playlist.objects.get(name=self.name)
                    return playlist.uri
                except Playlist.MultipleObjectsReturned:
                    results = sp.search(q=self._category + ':' + self.name, type=self._category, limit=1)
                    return results[self._category + 's']['items'][0]['uri']
                    
                
            else:
                raise ValueError("Invalid model provided.")

        except (Song.DoesNotExist, Artist.DoesNotExist, Playlist.DoesNotExist, Playlist.MultipleObjectsReturned):
            result = sp.search(q=self._category + ':' + self.name, type=self._category)  # search query

            if result and self._category + 's' in result and 'items' in result[self._category + 's']:
                spotify_obj = result[self._category + 's']['items'][0]
                uri = spotify_obj['uri']
                return uri
            else:
                raise Exception("No search results found for '{}' in category '{}'.".format(self.name, self._category))

    def track_list(self):
        """
        Load tracks on playlist OR artist top track into a dictionary

        :return: catalog: dictionary of songs with their uri
        """
        # cursor = db.execute(
        #     "SELECT name, danceability, energy, mode, valence, tempo, uri, key FROM song_props WHERE (name = ?)",
        #     (song_name,))
        # TODO if playlist is already in cache get info from cache!!
        if self._category == "playlist":
            track_list = []
            try:
                items = sp.playlist_items(self._uri)['items']
            except AttributeError:
                return None

            for item in items:
                #TODO in contactable search throws typeerror can't subscript Nonetype (item)

                if item.get('track'):
                    props = self.song_properties(track=(item['track']['name'], item['track']['uri']))
                    track_list.append((item['track']['name'], item['track']['uri']))
                else:
                    # Handle the case where 'track' is None
                    print("Track info is missing for this item:", item)
                
        elif self._category == "artist":
            track_list = []
            for item in sp.artist_top_tracks(self._uri)['tracks'][:50]:
                try:
                    artobj = Artist.objects.get(name=self.name)
                    props = self.song_properties(track = (item['name'], item['uri']))
                    songobj = Song.objects.get(name=props['name'])
                    artobj.tracks.add(songobj)

                except Artist.DoesNotExist:
                    print("artist object does not exist")
                    pass
                track_list.append((item['name'], item['uri']))
        else:
            return None

        catalog = {}
        for track in track_list:
            song, uri = track
            catalog.update({song: uri})

        return catalog

    @staticmethod
    def _properties_dict_gen(self, song_name, song_uri):
        """
        for use in song_properties generates song properties dict and adds to SongProps model
        """
        song_features = sp.audio_features(song_uri)
        track = sp.track(song_uri)
        popular = track['popularity']
        artist_info = sp.artist(track["artists"][0]["external_urls"]["spotify"])
        genre = str(artist_info['genres'])
        artist_name = artist_info['name']
        artist_uri = artist_info['uri']

        # Get or create the corresponding Artist object
        artist, _ = Artist.objects.get_or_create(name=artist_name, uri=artist_uri)


        song_props, _ = Song.objects.get_or_create(
            name=song_name,
            artist = artist,
            defaults={
                'danceability': song_features[0]['danceability'],
                'energy': song_features[0]['energy'],
                'mode': song_features[0]['mode'],
                'valence': song_features[0]['valence'],
                'tempo': song_features[0]['tempo'],
                'uri': song_uri,
                'key': song_features[0]['key'],
                'popularity': popular,
                'genre': genre,
            }
        )

        feature_dict = {
            'name': song_props.name,
            'artist': artist_name,
            'features': {
                'danceability': song_props.danceability,
                'energy': song_props.energy,
                'mode': song_props.mode,
                'valence': song_props.valence,
                'tempo': song_props.tempo / 100,  # divide tempo so it fits with other data
                'uri': song_props.uri,
                'key': song_props.key,
                'popularity': song_props.popularity,
                'genre': song_props.genre,
            }
        }
        return feature_dict

    def song_properties(self, track=None):
        """
        Try to get song info from the SongProps model if not available, create and store it using _properties_dict_gen()
        """
        try:
            if self._category == 'track' and track is None:
                song_name = self.name
                song_uri = self._uri
            else:
                song_name = track[0]
                song_uri = track[1]
        except AttributeError:
            song_name = track[0]
            song_uri = track[1]

        try:
            song_props = Song.objects.get(name=song_name)
            feature_dict = {
                'name': song_props.name,
                'artist': song_props.artist,
                'features': {
                    'danceability': song_props.danceability,
                    'energy': song_props.energy,
                    'mode': song_props.mode,
                    'valence': song_props.valence,
                    'tempo': song_props.tempo / 100,
                    'uri': song_props.uri,
                    'key': song_props.key,
                    'popularity': song_props.popularity,
                    'genre': song_props.genre,
                }
            }
            return feature_dict
        except Song.DoesNotExist:
            return self._properties_dict_gen(self=self, song_name=song_name, song_uri=song_uri)

    def multi_song_properties(self):
        trackdeets=[]
        for track in self.catalog.items():
            # print(track)
            props=self.song_properties(track=track)
            trackdeets.append(props)
        return trackdeets

    def _prep_feat_list_to_plot(self):
        """prepare feature values to plot on radial graph by scaling tempo and mode[major/minor] data
         to better match other metadata values"""
        feat_dict = self.song_properties()
        song_name = feat_dict['name']
        feat_dict = feat_dict['features']
        del feat_dict['uri']
        del feat_dict['key']
        print(feat_dict)
        categories = ["danceability", "energy", "mode", "valence", "tempo (divided by 100)"]
        categories = [*categories, categories[0]]
        feat_dict['tempo'] = feat_dict['tempo'] / 100  # scale down the tempo so that it is viewable with other data
        feat_dict['mode'] = feat_dict['mode'] / 5  # scale mode (either a value of 0 or 1)
        # so doesn't skew data, but differences still present
        # TODO experiment with above normalizations to make sure that the weighting works
        feat_list = []
        for key in feat_dict.keys():
            feat_list.append(feat_dict[key])
        feat_list = [*feat_list, feat_list[0]]
        return feat_list, song_name, categories

    def graph(self):
        """Plot sogn feature data on radial graph"""
        feat_list, song_name, categories = self._prep_feat_list_to_plot()
        fig = go.Figure(
            data=[
                go.Scatterpolar(r=feat_list, theta=categories, name=song_name),

            ],
            layout=go.Layout(
                title=go.layout.Title(text=self.name.upper()),
                polar={'radialaxis': {'visible': True}},
                showlegend=True
            )
        )
        pyo.plot(fig)

    # def overlay(self, other: object):
    #     """Overlay plots of feature data for 2 songs on radial graph"""
    #     feat_list, song_name, categories = self._prep_feat_list_to_plot()
    #     feat_list_2, other_name, _ = other._prep_feat_list_to_plot()
    #
    #     fig = go.Figure(
    #         data=[
    #             go.Scatterpolar(r=feat_list, theta=categories, name=song_name),
    #             go.Scatterpolar(r=feat_list_2, theta=categories, name=other_name),
    #
    #         ],
    #         layout=go.Layout(
    #             title=go.layout.Title(text="Song Features Comparison"),
    #             polar={'radialaxis': {'visible': True}},
    #             showlegend=True
    #         )
    #     )
    #     pyo.plot(fig)

    @staticmethod
    def _prep_feats_for_cosine_similarity(track=None):
        """Prepare song feature data for cosine similarity testing by scaling tempo, mode. Delete song uri, genre
        and keyt as they will not be used in the cosine similarity test"""
        song_name = track['name']
        feat_dict = track['features']
        del feat_dict['uri']
        del feat_dict['genre']
        del feat_dict['key']

        # categories = ["danceability", "energy", "mode", "valence", "tempo (divided by 100)", "key"]
        # categories = [*categories, categories[0]]
        feat_dict['tempo'] = feat_dict['tempo'] / 100  # scale down the tempo so that it is viewable with other data
        feat_dict['mode'] = feat_dict['mode'] / 4  # scale mode (either a value of 0 or 1)
        # feat_dict['key'] = feat_dict['key'] / 12  # scale key (range of 1 to 12) new highest value is 1

        # TODO experiment with above normalizations to make sure that the weighting works
        feat_list = []
        for key in feat_dict.keys():
            feat_list.append(feat_dict[key])
        return feat_list, song_name

    def similarness(self):
        """Take playlist songs from self.catalog get features for all songs, format data then load in pandas df
        ,run cosine similarity to generate edges, print node list and edge list to files 'spot_node.csv' and
        'spot_edge.csv' respectively, in a format usable by gephi. (I am currently altering to a format to be read by
        neo4j)
        """
        data = []
        data2 = []
        cleaned_data = []
        node_list = []
        for track in self.catalog.items():
            feat_list = self.song_properties(track=track)
            data.append(feat_list)
        # make a copy of your list of song data so you can remove the key, uri, and genre data for calculations
        # while keeping it for labeling your nodes
        data2 = copy.deepcopy(data)
        counter = 0
        for item in data2:
            feat_list, song_name = PlaylistClass._prep_feats_for_cosine_similarity(item)
            # feat_list.append(song_name)
            cleaned_data.append(feat_list)
            node_list.append({"id": song_name})
            counter += 1
        # print(cleaned_data)
        df = pd.DataFrame(cleaned_data,
                          columns=["danceability", "energy", "mode",
                                   "valence", "tempo (divided by 100)",
                                   "popularity"])

        # print('->')
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        link_list =[]
        with transaction.atomic():
            counter = 0
            for i, row in enumerate(cosine_similarity(df, df)):
                for index, element in enumerate(row):
                    if element > 0.97 and data[i]['name'] != data[index]['name']:
                        if counter % 20 == 0:
                            song_a = Song.objects.get(name=data[i]['name'])
                            song_b = Song.objects.get(name=data[index]['name'])
                            similarity_score = element

                            # Create a new SongRelationship instance and save it
                            link = SongRelationship.objects.get_or_create(
                                song_a=song_a,
                                song_b=song_b,
                                similarity_score=similarity_score
                            )
                            link_list.append({"source":link[0].song_a.name, "target": link[0].song_b.name, "value": link[0].similarity_score})
                            counter += 1
                        else:
                            counter += 1
                        

        # name = set()
        # outputDict = {"nodes":[], "links":[]}
        # outputHist = []
        # for i, row in enumerate(cosine_similarity(df, df)):
        #     for index, element in enumerate(row):
        #         ## CHUNK FOR EDGE_WEIGHT HISTOGRAM
        #         if index > i and element > 0.995: ## CAN VARY ELEMENT THRESHOLD
        #             outputHist.append(element)
        #         ## CHUNK FOR EDGE_WEIGHT HISTOGRAM

        #         if element > .97 and data[i]['name'] != data[index]['name']:
        #             if index>i: 
        #                 # selects only lower triangle of matrix but can't put it in larger
        #                 # if condition because it wouldn't allow the last node to be added.
        #                 # could make cleaner with thought but this works. 
        #                 # Need only lower triangle selected because this is an undirected graph
        #                 outputDict["links"].append({"source":i,"target":index,"value":log10(element*100)})

        #             if data[i]['name'] not in name:
        #                 name.add(data[i]['name'])
        #                 outputDict["nodes"].append({
        #                     "id":i,
        #                     "name":data[i]['name'],
        #                     "dance":data[i]['features']["danceability"],
        #                     "energy":data[i]['features']["energy"],
        #                     "mode":data[i]['features']["mode"],
        #                     "valence":data[i]['features']["valence"],
        #                     "bpm":data[i]['features']["tempo"],
        #                     "key":data[i]['features']["key"],
        #                     "popularity":data[i]['features']["popularity"],
        #                     "genre":data[i]['features']["genre"]
        #                 })
        # with open('spot_network.json', 'w') as outfile:
        #     dump(outputDict, outfile, indent=4)

        ## CHUNK FOR EDGE_WEIGHT HISTOGRAM
        # import matplotlib.pyplot as plt
        # plt.hist(outputHist, bins=100)
        # plt.show()
        ## CHUNK FOR EDGE_WEIGHT HISTOGRAM
        graph_data = {"nodes": node_list,"links": link_list}
        return graph_data

    def track_test(self, comp_song: object):
        # TODO write DOCSTRINGS! And fix code redundancy
        """tests how similar a single track is against all the tracks on the playlist.
        Generates edges between similar songs and prints data to comparison_edges.csv and comparison_node.csv"""

        data = []
        data2 = []
        cleaned_data = []
        track_props, song_name = Playlist._prep_feats_for_cosine_similarity(comp_song.song_properties())
        for track in self.catalog.items():
            feat_list = self.song_properties(track=track)
            data.append(feat_list)

        data2 = copy.deepcopy(data)
        for item in data2:
            feat_list, song_name = Playlist._prep_feats_for_cosine_similarity(item)
            # feat_list.append(song_name)
            cleaned_data.append(feat_list)

        # print(cleaned_data)
        # print(track_props)
        df = pd.DataFrame(cleaned_data,
                          columns=["danceability", "energy", "mode",
                                   "valence", "tempo (divided by 100)",
                                   "popularity"])

        df2 = pd.DataFrame(track_props).T
        # print('->')
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        df2.columns = ["danceability", "energy", "mode",
                       "valence", "tempo (divided by 100)",
                       "popularity"]

        edge_weight = ["Weight"]
        source = ["Source"]
        name = ["Label"]
        num_id = ["ID"]
        target = ["Target"]
        dance = ["danceability"]
        energy = ["energy"]
        mode = ["mode"]
        valence = ["valence"]
        bpm = ["tempo"]
        key = ["key"]
        popularity = ["popularity"]
        genre = ["genre"]

        for i, row in enumerate(cosine_similarity(df, df2)):
            for index, element in enumerate(row):
                if element > .97 and data[i]['name'] != data[index]['name']:
                    edge_weight.append(element)
                    source.append(i)
                    target.append(index)
                    if data[i]['name'] not in name:
                        name.append(data[i]['name'])
                        num_id.append(i)
                        dance.append(data[i]['features']["danceability"])
                        energy.append(data[i]['features']["energy"])
                        mode.append(data[i]['features']["mode"])
                        valence.append(data[i]['features']["valence"])
                        bpm.append(data[i]['features']["tempo"])
                        key.append(data[i]['features']["key"])
                        popularity.append(data[i]['features']["popularity"])
                        genre.append(data[i]['features']["genre"])

        edge = list(zip(source, target, edge_weight))
        node = list(zip(name, num_id, dance, energy, mode, valence, bpm, key, popularity, genre))
        with open("comparison_edge.csv", 'w') as edges:
            for source, target, weight in edge:
                print("{},{},{}".format(source, target, weight), file=edges)

        with open("comparison_node.csv", 'w') as nodes:
            for n_id, label, dance, energy, mode, valence, bpm, key, popularity, genre in node:
                print("{},{},{},{},{},{},{},{},{},{}"
                      .format(n_id, label, dance, energy, mode, valence, bpm, key, popularity, genre), file=nodes)
        return edge, node


class PlaylistClass(SpotSuper):
    def __init__(self, name, model='playlist'):
        super().__init__(name=name, category="playlist", model=model)
        # self.mean_feats = self.mean_playlist_feats(self.catalog)
        self.playobj = self.get_or_create_playlist()
    
    def get_or_create_playlist(self, uri=None):
        
        if uri:
            pl_obj = sp.playlist(uri)
        else:
            pl_obj = sp.playlist(self._uri)
        name = pl_obj['name']
        owner_obj = pl_obj['owner']
        follows = pl_obj['followers']['total']
        desc = pl_obj['description']
        cover = pl_obj['images'][0]['url']
        processed_owner = (owner_obj['display_name'],owner_obj['uri'])
        # Attempt to get the Playlist object by name
        try:
            playlist = Playlist.objects.get(name=name)
        except Playlist.DoesNotExist:
            playlist = Playlist.objects.create(
                name=name,
                owner=processed_owner,
                follow_count=follows,
                description=desc,
                cover_image=cover,
                uri=self._uri
            )
        catalog = self.track_list()
        for song in catalog:
            songobj = Song.objects.get(name=song)
            playlist.songs.add(songobj)
            art = songobj.artist
            art.playlists_featured_on.add(playlist)
        return playlist

    def network_plot(self):
        edge, node = self.similarness()
        print(edge)
        print(node)
    
    def contactable_search(self):
    # TODO bug same playlists are being saved multiple times
        all_results = sp.search(q=self._category + ':' + self.name, type=self._category, limit=30)
        result_list = []
        for item in all_results[self._category + 's']['items']:
            playlist_name = item['name']
            playlist = Playlist.objects.get(name=playlist_name)

            if '@' in playlist.description: 
                result_list.append((playlist_name, item['owner']['display_name'], item['uri']))
        return result_list
        




class CrawlClass(PlaylistClass):
    def __init__(self, name):
        super().__init__(name=name, model='thicc')
        self.name = name
        self._uri = self.search()

    def crawling(self):
                    
        categories = sp.categories(limit=50)
        all_lists = []
        for category in categories:
            playlists = sp.category_playlists(category_id=category, limit=50)
            all_lists.append(playlists)
        
        feat_lists = sp.featured_playlists(limit=50)
        for list in feat_lists:
            all_lists.append(list)
        

        bacth_list = [
            "A playlist for september",
            "Fixing your shitty music taste",
            "axel's release radar",
            "The smoker's club pregame",
            "songs that are perfect",
            "talking to the moon",
            "save this for a rainy day",
            "songs that ar eway to dangerous to listen to alone",
            "hidden gems.",
            "songs that slap",
            "sonder szn",
            "Indie Waver",
            "Night Trip",
            "Vibes for the Day",
            "Morning Right",
            "Songs you forgot about",
            "summer aux",
            "let me float",
            "therapeutic",
            "frequency",
            "dancing in the kitchen",
            "oh i love it and i hate it at the same time",
            "true that I saw her hair like the branch of a tree",
            "In the back of my mind, you died",
            "New Finds",
            "a song a day",
            "these things run though my mind while I'm by myself",
            "On Repeat",
            "a walk with my earbuds in",
            "leaning against a train window",
            "Anxiety Free :)",
            "life is peaking, but in a different way",
            "Fall 2022",
            "lost love...",
            "i need him like water",
            "songs you need to hear at least once",
            "can I bite your tongue like my bad habit?",
            "rainy day songs...",
            "Good Morning",
            "Mr. Forgettable",
            "In my own world",
            "Dance walking",
            "Rancid Eddie - Dry",
            "Just a Little While",
            "6 cars & a grizzly bear",
            "Miserable Man",
            "Background Music",
            "Good Looking - Suki Waterhouse",
            "I wanna run away and live in the woods",
            "sitting watching the world end",
            "3 am driving into the morning",
            "the saddest songs i know",
            "Notion - The Rare Occasions",
            "depollute my pretty baby...",
            "i think i like when it rains",
            "iOS 15 Background Sounds",
            "found - zach webb",
            "beautiful love songs",
            "mending a broken heart",
            "Silk Sonic",
            "ur a frog on a leaf in the garden",
            "Vibe Test Songs",
            "Sumer Rotation",
            "Running Up That Hill - Kate Bush",
            "Floating amongst the clouds",
            "You... You is the Main Character",
            "Pope Is a Rockstar",
            "Memoir #2",
            "Falling in love back in the day",
            "Where My Rosemary Goes",
            "i love you so...",
            "I'm going back to 505",
            "as it was",
            "songs about the moon",
            "It's Called: Freefall",
            "Lost",
            "Incomplete & Depressed",
            "Songs that are not rap that you'll love",
            "Thinking about life while zoning out of reality",
            "Actually, I Have Been Broken All Along",
            "Falling Apart",
            "Songs that make you leave your body",
            "Rage, Give in to your anger!",
            "POV I Got Passed The AUX",
            "Pov I'm gonna fist fight a demon",
            "Bass to Blow Out Your CAr Spaeakers",
            "Study Lofi For MAX Focus",
            "Straight Motivation",
            "The Hottest love has the coldest end",
            "Late Night Drive",
            "Underrated Gems",
            "Viral Trending Tik Tok Tracks",
            "MOSHPIT",
            "Phonk Beats and high speeds",
            "Gym Apex Monster Playlist",
            "Songs That Make You Feel Good",
            "Bedroom Lovin",
            "Summer Feel Good Vibes",
            "Falling in Love",
            "Party Vibes 24/7",
            "Drill Elite",
            "Best of Aesthetic Rap",
            "Chill rap vibes",
            "Intense Hyperpop",
            "Beauty in the Darkness",
            "Underrated Rap & Hip Hop",
            "We Had The Right Love at The Wrong Time",
            "Chill vibe songs",
            "Slow & Reverb to Perfection",
            "Be the endergy you want to attract",
            "Masked Mortal New Music Friday",
            "Best of Anime Songs",
            "Euphoria Main Character Vibes",
            "Dancing in the Kitchen at 3am",
            "The Bun 91.3 Official Playlist",
            "Backroads In The Key Of Indie",
            "The Cover Song Aesthetic",
            "The Happiness Aesthetic",
            "the Velvet series",
            "smooth as silk",
            "a playlist for your crush",
            "Raindrop Feels",
            "Fall Flannels",
            "late. night. listens.",
            "Fall Yellow Day Dreams",
            "The State of (Indie)ana",
            "Heat Check",
            "I'd Do Bad Things",
            "Daily Rotation",
            "Feels Like A Dream",
            "Songs I Like That You'll Like Too",
            "matcha & granola",
        ]
        all_lists.append(bacth_list)
        count = 0
        for item in all_lists:
            thisinstance = PlaylistClass(name=item['name'])
            count += 1
            print("batch " + count + " done")



class ArtistClass(SpotSuper):
    def __init__(self, name):
        super().__init__(name=name, category="artist", model='artist')

class TrackClass(SpotSuper):
    def __init__(self, name):
        super().__init__(name=name, category="track",model='track')


if __name__ == '__main__':
    #  below is sample code analyzing the POLLEN playlist and how Passionfruit relates and its contents
    passionfruit = TrackClass("Passionfruit")
    passionfruit.song_properties()
    #
    #     dum_surfer = Track("dum surfer")
    #     # dum_surfer.song_properties()
    #
    #     onetake = Track("onetake interlude")
    #     # onetake.song_properties()
    #     sad = Track("cementality")
    #     sad.song_properties
    #     # onetake.overlay(passionfruit)
    pollen = PlaylistClass("POLLEN")
    pollen.similarness()
    # pollen.track_test(passionfruit)
    #     # pollen.network_plot()
    #
    drake = ArtistClass("Drake")
    # drake.track_test(passionfruit)
#     # dork = SpotSuper("dork", "track", conn)
#     # dork.overlay(dum_surfer)
#     # i_might = SpotSuper("I Might slip away if I don't feel nothing", "track", conn)
#     # thot = SpotSuper("Thot Tactics", "track", conn)
#     # i_might.overlay(thot)
#     conn.close()

# TODO improve searching mechanic so smaller artists will be able to find their stuff
# TODO  I should combine key and mode for a label on the nodes