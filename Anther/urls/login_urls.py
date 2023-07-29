from django.urls import path
import Anther.views.login_views as views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('login/register/',views.registration_view, name='register'),
    path('mailing_list/', views.add_to_mailing_list, name='add_to_mailing_list')
]