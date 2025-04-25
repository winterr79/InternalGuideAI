"""
View functions for the chatbot application.

Handles HTTP requests, processes user messages, interacts with the Gemini API,
and returns responses to the frontend.
"""

import json
import os
import requests
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseServerError
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import ChatLog

# The path to our product data JSON file
PRODUCT_DATA_PATH = os.path.join(settings.PROJECT_ROOT, 'data', 'product_info.json')

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
    
    Handles the core chatbot functionality:
    1. Receives user message
    2. Identifies relevant product data (if any)
    3. Retrieves conversation history
    4. Calls Gemini API with appropriate context
    5. Logs the interaction
    6. Returns response to the frontend
    """
    try:
        # Parse the incoming JSON data
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        # Initialize session history if not present
        if 'chat_history' not in request.session:
            request.session['chat_history'] = []
        
        # Get existing history (limited to last 5 exchanges for context window)
        history = request.session['chat_history'][-5:] if request.session['chat_history'] else []
        
        # Get product information from JSON file
        product_info = load_product_data()
        if not product_info:
            return JsonResponse({'error': 'Product data unavailable'}, status=500)
            
        # Identify which product (if any) the user is asking about
        product_context, grounding_data = identify_product_context(user_message, product_info)
        
        # Add the user's message to history
        history.append({"role": "user", "content": user_message})
        
        # Call Gemini API with appropriate context
        if grounding_data:
            # We have relevant product info, use it as grounding
            response = call_gemini_api(user_message, history, grounding_data, product_context)
        else:
            # No specific product identified, use general response with available products
            product_names = list(product_info.keys())
            response = generate_out_of_scope_response(product_names)
        
        # Add bot response to history
        history.append({"role": "assistant", "content": response})
        
        # Update session with new history (keep last 10 messages maximum)
        request.session['chat_history'] = history[-10:]
        request.session.modified = True
        
        # Log the interaction in the database
        ChatLog.objects.create(
            session_key=request.session.session_key,
            user_input=user_message,
            bot_response=response,
            product_context=product_context
        )
        
        return JsonResponse({'response': response})
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        # Log the error (in a production app, use proper logging)
        print(f"Error processing message: {str(e)}")
        return JsonResponse({'error': 'An error occurred processing your request'}, status=500)

def load_product_data():
    """
    Loads product information from the JSON file.
    
    Returns a dictionary of product data or None if the file cannot be loaded.
    """
    try:
        with open(PRODUCT_DATA_PATH, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"Error loading product data: {str(e)}")
        return None

def identify_product_context(user_message, product_info):
    """
    Identifies which product (if any) the user is asking about based on simple keyword matching.
    
    Returns the product name and its associated data, or (None, None) if no match is found.
    """
    # Convert to lowercase for case-insensitive matching
    message_lower = user_message.lower()
    
    # Check each product name and common variations/keywords
    for product_name, product_data in product_info.items():
        # Create variations of the product name for matching
        name_lower = product_name.lower()
        keywords = [name_lower]
        
        # Add word-by-word checks (e.g., "busy board" for "Wooden Busy Board")
        keywords.extend([word.lower() for word in product_name.split() if len(word) > 3])
        
        # Check for matches
        if any(keyword in message_lower for keyword in keywords):
            return product_name, product_data
    
    # No specific product identified
    return None, None

def call_gemini_api(user_query, history, grounding_data, product_name):
    """
    Calls the Google Gemini API with the user query, conversation history, 
    and product-specific grounding data.
    
    Returns the model's response text or an error message if the API call fails.
    """
    try:
        # Format the grounding data as a string
        grounding_text = format_product_data(product_name, grounding_data)
        
        # Construct the API request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{
                        "text": construct_prompt(user_query, history, grounding_text, product_name)
                    }]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        # Set up API request headers with the API key
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.GEMINI_API_KEY
        }
        
        # Make the API request
        response = requests.post(
            settings.GEMINI_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=10  # Timeout after 10 seconds
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response JSON
        response_data = response.json()
        
        # Extract the generated text from the response
        if 'candidates' in response_data and response_data['candidates']:
            # Get text from the first candidate's first part
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'text' in part:
                        return part['text'].strip()
        
        # If we couldn't extract text using the expected structure
        return "I'm having trouble generating a response right now. Please try again."
        
    except requests.RequestException as e:
        print(f"API request error: {str(e)}")
        return "Sorry, I couldn't connect to my knowledge source. Please try again in a moment."
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing API response: {str(e)}")
        return "I received an unexpected response format. Please try asking again."
    except Exception as e:
        print(f"Unexpected error calling Gemini API: {str(e)}")
        return "Something went wrong while processing your request. Please try again."

def format_product_data(product_name, product_data):
    """
    Formats the product data dictionary into a readable string for the API prompt.
    """
    if not product_data:
        return ""
        
    # Build a formatted string with all product attributes
    result = f"Product: {product_name}\n"
    for key, value in product_data.items():
        formatted_key = key.replace('_', ' ').title()
        result += f"{formatted_key}: {value}\n"
    
    return result

def construct_prompt(user_query, history, grounding_text, product_name):
    """
    Constructs the complete prompt for the Gemini API, including system instructions,
    grounding data, conversation history, and the user's current query.
    """
    # System instructions
    system_prompt = (
        "You are a helpful and professional Internal Guide AI assistant for Sensory Souk. "
        "You specialize in providing accurate information about sensory products. "
        "You must ONLY answer based on the provided product information below. "
        "If you don't know something or it's not in the provided information, "
        "say you don't have that information. Never make up details. "
        "Be concise, friendly, and helpful."
    )
    
    # Format conversation history
    history_text = ""
    if history and len(history) > 1:  # Only include if there's actual back-and-forth
        history_text = "Here's our conversation so far:\n"
        for entry in history[:-1]:  # Exclude current user query (added separately)
            role = "User" if entry["role"] == "user" else "Assistant"
            history_text += f"{role}: {entry['content']}\n"
    
    # Add the current query
    current_query = f"User's current question: {user_query}"
    
    # Build full prompt
    full_prompt = f"{system_prompt}\n\n"
    
    if grounding_text:
        full_prompt += f"PRODUCT INFORMATION:\n{grounding_text}\n\n"
    
    if history_text:
        full_prompt += f"{history_text}\n"
    
    full_prompt += current_query
    
    return full_prompt

def generate_out_of_scope_response(product_names):
    """
    Generates a response for when the user asks about a topic outside of our knowledge base.
    
    Lists the products we do have information about.
    """
    product_list = ", ".join(product_names[:-1]) + (", and " + product_names[-1] if len(product_names) > 1 else product_names[0])
    
    return (
        f"I'm sorry, I don't have information about that. Currently, I can only provide details about the following products: "
        f"{product_list}. How can I help you with any of these products?"
    )