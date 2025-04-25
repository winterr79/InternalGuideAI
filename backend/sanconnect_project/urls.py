"""
Project-level URL configuration.

This file defines the high-level URL patterns for the entire project,
including admin routes and inclusion of app-specific URL patterns.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Include all URLs from the chatbot app
    path('', include('chatbot_app.urls')),
]
