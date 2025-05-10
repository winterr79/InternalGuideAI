"""
View functions for the chatbot application.

Handles HTTP requests, processes user messages, implements the RAG pipeline,
interacts with the Gemini API, and returns responses to the frontend.
"""

import json
import os

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
import google.generativeai as genai

import numpy as np
from chatbot_app.management.commands.update_vectordb import VectorStore
from .models import ChatLog

# Initialize Google Generative AI with API key
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Google Generative AI: {str(e)}")

# Initialize VectorStore (do this once when module loads)
try:
    vector_store_instance = VectorStore()
    if not vector_store_instance.index or not vector_store_instance.chunks_metadata:
        print("Warning: VectorStore initialized but index or metadata is empty")
except Exception as e:
    print(f"Error initializing VectorStore: {str(e)}")
    vector_store_instance = None

@ensure_csrf_cookie
def chat_view(request):
    """
    Renders the main chat interface page.
    
    Loads the HTML template and ensures a CSRF token is set.
    """
    return render(request, 'chatbot_app/chat_interface.html')

@require_http_methods(["POST"])
def process_message(request):
    """
    Processes incoming chat messages from the user.
    
    Implements the RAG (Retrieval-Augmented Generation) pipeline:
    1. Receives user message
    2. Generates embedding for the user query
    3. Retrieves relevant chunks from the vector store
    4. Augments the user query with the retrieved context
    5. Sends the augmented query to the Gemini API
    6. Updates conversation history and logs the interaction
    7. Returns response to the frontend
    """
    try:
        # Parse the incoming JSON data
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        # Validate user input
        if not user_message:
            return JsonResponse({'error': 'Empty message'}, status=400)
        
        # Initialize session history if not present
        if 'chat_history' not in request.session:
            request.session['chat_history'] = []
        
        # Get existing history (limited to last 5 exchanges for context window)
        history = request.session['chat_history'][-5:] if request.session['chat_history'] else []
        
        # Check if VectorStore is available
        if not vector_store_instance or not vector_store_instance.index:
            return JsonResponse({
                'response': "I'm sorry, my knowledge database is currently unavailable. Please try again later."
            })
        
        # RAG Pipeline Implementation
        retrieved_context_text = ""
        try:
            # 1. Generate embedding for user query
            query_embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=user_message,
                task_type="retrieval_query"
            )
            query_embedding = query_embedding_result['embedding']
            
            # 2. Search FAISS for similar chunks (get top 3 results)
            k = 3  # Number of chunks to retrieve
            query_vector_np = np.array([query_embedding], dtype='float32')
            distances, indices = vector_store_instance.index.search(query_vector_np, k)
            
            # 3. Retrieve the text of the top chunks
            relevant_chunks = []
            for idx in indices[0]:  # indices[0] because we only have one query vector
                if idx != -1 and idx < len(vector_store_instance.chunks_metadata):  # -1 means no match
                    chunk_metadata = vector_store_instance.chunks_metadata[idx]
                    relevant_chunks.append({
                        'text': chunk_metadata['text'],
                        'url': chunk_metadata['url'],
                        'title': chunk_metadata['title']
                    })
            
            # 4. Format retrieved chunks as context
            if relevant_chunks:
                context_parts = []
                for i, chunk in enumerate(relevant_chunks, 1):
                    source_info = f"Source: {chunk['title']} ({chunk['url']})"
                    context_parts.append(f"[Chunk {i}]\n{chunk['text']}\n{source_info}")
                
                retrieved_context_text = "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            # Continue with empty context if retrieval fails
            retrieved_context_text = ""
        
        # Prepare for Gemini API call
        
        # 1. Define system instruction
        system_instruction = (
            "You are an expert internal assistant for Sensory Souk. Your role is to answer "
            "employee questions about Sensory Souk products accurately and helpfully. "
            "Strictly base your answers on the 'Relevant Information from Website' provided below. "
            "If the information isn't mentioned in the provided context, state that you don't have "
            "that specific detail from the website content. Do not make up information or answer "
            "questions outside the scope of the provided product details. "
            "Be concise, friendly, and helpful. If asked a general greeting, respond politely."
        )
        
        # 2. Format conversation history for Gemini API
        api_call_history = []
        for entry in history:
            api_call_history.append({
                "role": entry["role"],
                "parts": [{"text": entry["content"]}]
            })
        
        # 3. Prepare user's current message (with context if available)
        current_user_message = user_message
        if retrieved_context_text:
            current_user_message = (
                f"Use the following information to answer the question:\n"
                f"---BEGIN INFORMATION---\n{retrieved_context_text}\n---END INFORMATION---\n\n"
                f"Question: {user_message}"
            )
        
        # Call Gemini API
        try:
            # Initialize the model with system instruction
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                system_instruction=system_instruction
            )
            
            # Start/continue chat with history
            chat = model.start_chat(history=api_call_history)
            
            # Send the current user message (with context if available)
            response = chat.send_message(current_user_message)
            
            # Extract the text response
            bot_response = response.text
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            bot_response = "I'm having trouble accessing my knowledge at the moment. Please try again shortly."
        
        # Update history with user message and bot response
        request.session['chat_history'].append({
            "role": "user", 
            "content": user_message  # Store original user message, not augmented
        })
        request.session['chat_history'].append({
            "role": "model", 
            "content": bot_response
        })
        
        # Limit history size (keep last 10 exchanges)
        if len(request.session['chat_history']) > 20:  # 10 exchanges = 20 messages
            request.session['chat_history'] = request.session['chat_history'][-20:]
            
        request.session.modified = True
        
        # Log the interaction in the database
        ChatLog.objects.create(
            session_key=request.session.session_key or "anonymous",
            user_input=user_message,
            bot_response=bot_response,
            product_context="RAG" if retrieved_context_text else "No context"
        )
        
        return JsonResponse({'response': bot_response})
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        # Log the error
        print(f"Error processing message: {str(e)}")
        return JsonResponse({'error': 'An error occurred processing your request'}, status=500)