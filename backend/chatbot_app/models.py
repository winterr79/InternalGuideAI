"""
Database models for the chatbot application.

Defines the database schema for storing chat logs and related data.
"""

from django.db import models

class ChatLog(models.Model):
    """
    Model for storing conversation logs between users and the chatbot.
    
    Tracks the session, messages, timestamps, and product context for each interaction.
    """
    session_key = models.CharField(max_length=40, blank=True, null=True)
    user_input = models.TextField()
    bot_response = models.TextField()
    product_context = models.CharField(max_length=100, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        """String representation of the chat log entry"""
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {self.user_input[:30]}..."
    
    class Meta:
        """Meta options for the ChatLog model"""
        ordering = ['-timestamp'] 