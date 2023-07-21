from django.urls import path
import views.team_views as views

urlpatterns = [
    path('team/', views.team_view, name='team'),
]