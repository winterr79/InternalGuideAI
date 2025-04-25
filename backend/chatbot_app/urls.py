"""
URL patterns for the chatbot application.

Defines the route mapping between URL paths and view functions for the chatbot app.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Main chat interface URL - serves as the homepage
    path('', views.chat_view, name='chat_interface'),
    # API endpoint for handling chat messages
    path('api/chat/', views.process_message, name='process_message'),
]