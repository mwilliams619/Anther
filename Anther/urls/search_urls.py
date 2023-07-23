from django.urls import path
import Anther.views.search_views as views

urlpatterns = [
    path('search/', views.search_view, name='search-view'),
]