from django.urls import path
import Anther.views.home_views as views

urlpatterns = [
    path('', views.home_view, name='home'),
]