from django.urls import path
import Anther.views.login_views as views

urlpatterns = [
    path('login/', views.login_view, name='login'),
]