from django.urls import path
import views.login_views as views

urlpatterns = [
    path('login/', views.login_view, name='login'),
]