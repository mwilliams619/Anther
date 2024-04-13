from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.db import connection
from django.shortcuts import render
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
# from instagram_private_api import Client as InstagramClient
# from tiktok_api import TikTokAPI
# import googleapiclient.discovery

from Anther.models import MoneySpent, SongMetrics, ArtistFollower

def metrics_view(request):

    money_spent = MoneySpent.objects.filter(...) 
    song_data = SongMetrics.objects.filter(...) 
    followers_data = ArtistFollower.objects.filter(...)

    return render(request, 'metrics.html', {
        "money_spent": money_spent,
        "songData": song_data,
        "followersData": followers_data
    })

@api_view(['GET']) 
def get_metrics_data(request):

    # Get data from models
    money_spent = MoneySpent.objects.all() \
                    .order_by('spent_at') \
                    .values('spent_at', 'amount')

    song_streams = SongMetrics.objects.all() \
                    .order_by('last_updated') \
                    .values('last_updated', 'spotify_streams')
    
    followers = ArtistFollower.objects.all()\
                    .order_by('date')\
                    .values('date', 'spotify_followers')
    
    # Package data            
    metrics_data = {
        "money": list(money_spent), 
        "songs": list(song_streams),
        "followers": list(followers), 
    }
    
    return Response(metrics_data)

# def metrics_view(request):

#     return render(request, 'app/metrics.html') 

def data_entry_view(request):
  return render(request, 'fin_dash.html')

class ArtistFollowerSerializer(serializers.ModelSerializer):
   
    class Meta:
        model = ArtistFollower
        fields = '__all__'

class MoneySpentSerializer(serializers.ModelSerializer):

    class Meta: 
        model = MoneySpent
        fields = '__all__'

class SongMetricsSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = SongMetrics
        fields = '__all__'

@api_view(['POST'])  
def submit_form(request):
    data = request.data
    model_name = data['model']
    
    if model_name == 'ArtistFollower':
        serializer = ArtistFollowerSerializer(data=data['fields']) 
    elif model_name == 'MoneySpent':
        serializer = MoneySpentSerializer(data=data['fields'])
    elif model_name == 'SongMetrics':
        serializer = SongMetricsSerializer(data=data['fields'])

    if serializer.is_valid():
        serializer.save()
        return Response(status=201)
    
    return Response(serializer.errors, status=400)

@api_view(['GET'])
def get_model_fields(request, model_name):

    if model_name == 'SongMetrics':
        model = SongMetrics
    elif model_name == 'ArtistFollower':  
        model = ArtistFollower
    elif model_name == 'SongMetrics':
        model = SongMetrics
    # and so on

    fields = []

    for field in model._meta.get_fields():
        field_data = {
            'name': field.name,
            'type': field.get_internal_type() 
        }
        
        fields.append(field_data)

    return Response(fields)