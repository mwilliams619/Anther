from django.urls import path
import Anther.views.fin_dash_views as views

urlpatterns = [
    path('dashboard/', views.data_entry_view, name='metrics entry'),
    path('dashboard/submit-form/', views.submit_form, name='submit_form'),
    path('dashboard/<model_name>', views.get_model_fields, name='get model fields'),
]