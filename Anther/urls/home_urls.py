from django.urls import path
import views.home_views as views

urlpatterns = [
    path('', views.home_view, name='home'),
]