from django.urls import path
import Anther.views.team_views as views

urlpatterns = [
    path('team/', views.team_view, name='team'),
]