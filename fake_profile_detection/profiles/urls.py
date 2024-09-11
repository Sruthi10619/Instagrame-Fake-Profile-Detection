# profiles/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Define URL patterns for profile-related views
    path('', views.index, name='index'),  # Example: Homepage view
    path('profile/<str:username>/', views.profile_detail, name='profile_detail'), 
    
]
