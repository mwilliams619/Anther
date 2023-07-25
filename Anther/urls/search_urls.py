from django.urls import path
import Anther.views.search_views as views

urlpatterns = [
    path('search/', views.search_view, name='search-view'),
    path('search/results', views.search_results, name='search-results'),
    path('search/link_find', views.link_find, name='song-links')
]