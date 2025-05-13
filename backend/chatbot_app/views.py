"""
View functions for the chatbot application.

Handles HTTP requests, processes user messages, implements the RAG pipeline,
interacts with the Gemini API, and returns responses to the frontend.
"""

import json
import os
import re
from collections import defaultdict
import textwrap

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from chatbot_app.management.commands.update_vectordb import VectorStore
from .models import ChatLog

# Initialize Gemini API and VectorStore
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Google Generative AI: {str(e)}")

try:
    vector_store_instance = VectorStore()
    if not vector_store_instance.index or not vector_store_instance.chunks_metadata:
        print("Warning: VectorStore initialized but index or metadata is empty")
except Exception as e:
    print(f"Error initializing VectorStore: {str(e)}")
    vector_store_instance = None

@ensure_csrf_cookie
def chat_view(request):
    return render(request, 'chatbot_app/chat_interface.html')

def extract_potential_product_name(user_message):
    """
    Extract potential product names from user message using various patterns.
    Returns the first potential product name found or None.
    """
    # Check for quoted product names first (highest confidence)
    quoted_match = re.search(r'["\']([^"\']+)["\']', user_message)
    if quoted_match:
        product_name = quoted_match.group(1).strip()
        if len(product_name) > 3 and not re.fullmatch(r'[?.!,;:]+', product_name) and product_name.lower() not in ["the product", "a product", "this item"]:
            return product_name

    # Check for product mentions after common phrases
    patterns_suffix = [
        # More specific first
        r"what are the benefits of (.+)",
        r"what is the age range for (.+)",
        r"are there any warnings for (.+)",
        r"describe the (.+)",
        r"tell me about (.+)",
        r"details for (.+)",
        r"info on (.+)",
        r"features of (.+)",
        r"specifications for (.+)",
        r"how to use (.+)",
        r"material of (.+)",
        r"what is (.+)",
        r"compare (.+) and (.+)",  
        r"how does (.+) compare to (.+)",  
        r"difference between (.+) and (.+)",  
        r"can you describe (.+)",
        r"do you have information on (.+)",
        r"learn about (.+)",
        r"show me details about (.+)",
        r"give me details on (.+)",
    ]
    
    for pattern in patterns_suffix:
        matches = re.search(pattern, user_message.lower())
        if matches:
            # Special handling for comparison patterns with two groups
            if "and" in pattern or "to" in pattern or "between" in pattern:
                if len(matches.groups()) == 2:
                    return [matches.group(1).strip(), matches.group(2).strip()]
            
            # Single product pattern
            product_candidate = matches.group(1).strip().rstrip('.?!,:;')
            if len(product_candidate) > 3 and product_candidate.lower() not in ["the product", "a product", "this item", "this", "that", "it", "them"]:
                return product_candidate

    # Fallback: If the message is short and seems like just a product name
    words_in_message = user_message.lower().split()
    common_question_words = ["what", "how", "who", "when", "where", "why", "can", "is", "are", "do", "does", "tell", "describe", "show", "list", "give", "find"]
    
    if len(words_in_message) <= 7 and not any(word in common_question_words for word in words_in_message):
        product_candidate = user_message.strip().rstrip('?').strip()
        if (len(product_candidate.split()) >= 1 or len(product_candidate) > 5) and product_candidate.lower() not in ["it", "this", "that", "them", "the product", "a product", "this item", "something", "anything"]:
            # Check if it's likely a category by looking for common category keywords if it's multiple words
            category_keywords = ["category", "collection", "group", "type", "kind"]
            is_likely_category_query = any(keyword in product_candidate.lower() for keyword in category_keywords)
            if not is_likely_category_query or len(words_in_message) == 1:  
                if len(product_candidate) > 3:  
                    return product_candidate
    
    return None

def analyze_query_intent(user_message, product_name=None):
    """
    Analyze the user's query to determine the specific intent and information needs.
    Returns a dict with intent_type and any additional metadata.
    """
    user_msg_lower = user_message.lower()
    
    # Check if it's a comparison query first (two products)
    if isinstance(product_name, list) and len(product_name) == 2:
        return {
            "intent_type": "comparison",
            "products": product_name,
            "aspects": extract_comparison_aspects(user_msg_lower)
        }
    
    # Check for category listing intent
    category_patterns = [
        r"list all (.+)",
        r"show all (.+)",
        r"what (.+) do you have",
        r"what are the (.+) available",
        r"types of (.+)",
        r"categories of (.+)"
    ]
    
    for pattern in category_patterns:
        match = re.search(pattern, user_msg_lower)
        if match:
            category = match.group(1).strip()
            if category.endswith("s") and len(category) > 4:  
                return {
                    "intent_type": "category_listing",
                    "category": category
                }
    
    # If we have a product name, determine the specific information need
    if product_name and not isinstance(product_name, list):
        # Specific product intents
        intents = {
            "usage": ["how to use", "how do you use", "instructions", "donning", "steps", "wear", "put on", "apply"],
            "benefits": ["benefit", "advantage", "help with", "good for", "pros"],
            "materials": ["material", "made of", "composition", "fabric", "constructed"],
            "features": ["feature", "what does it do", "capabilities", "what can", "function"],
            "warnings": ["warning", "caution", "side effect", "risk", "danger", "avoid"],
            "sizing": ["size", "measurement", "dimension", "fit", "how big", "how small"],
            "conditions": ["condition", "ideal for", "helps with", "treatment", "therapy", "disorder"],
            "research": ["research", "study", "evidence", "proven", "tested", "efficacy"],
            "age_range": ["age", "children", "adults", "toddler", "infant", "teenager", "senior"],
            "pricing": ["price", "cost", "how much", "expensive", "discount", "sale"],
            "availability": ["available", "in stock", "out of stock", "backorder", "shipping"]
        }
        
        for intent, keywords in intents.items():
            if any(keyword in user_msg_lower for keyword in keywords):
                return {
                    "intent_type": "product_specific",
                    "product": product_name,
                    "information_need": intent
                }
        
        # General product info (no specific aspect identified)
        return {
            "intent_type": "product_general",
            "product": product_name
        }
    
    # General query (not product or category specific)
    return {
        "intent_type": "general",
        "query": user_message
    }

def extract_comparison_aspects(query_text):
    """Extract specific aspects to compare between products"""
    aspects = []
    comparison_aspects = [
        "price", "cost", "material", "size", "feature", "benefit", 
        "usage", "durability", "quality", "weight", "dimension",
        "age", "condition", "effectiveness"
    ]
    
    for aspect in comparison_aspects:
        if aspect in query_text or f"{aspect}s" in query_text:
            aspects.append(aspect)
    
    return aspects if aspects else ["general"]  

def perform_rag_retrieval(query_text, k_value, filters=None):
    """
    Perform RAG retrieval for a given query text with optional filters.
    
    Args:
        query_text: The query text to search for
        k_value: Number of chunks to retrieve
        filters: Dict with optional filters like {'url_contains': 'products/spio'}
    
    Returns:
        List of retrieved chunk data
    """
    try:
        query_embedding_result = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_result['embedding']
        query_vector_np = np.array([query_embedding], dtype='float32')
        
        if not vector_store_instance or not vector_store_instance.index:
            print("Vector store not available for RAG retrieval.")
            return []

        # Retrieve a larger set initially to allow for filtering
        k_search = min(k_value * 3, 30)  
        distances, indices = vector_store_instance.index.search(query_vector_np, k_search)
        
        retrieved_chunks_data = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(vector_store_instance.chunks_metadata):
                chunk_metadata = vector_store_instance.chunks_metadata[idx]
                
                # Apply filters if specified
                if filters:
                    if 'url_contains' in filters and filters['url_contains'] not in chunk_metadata['url'].lower():
                        continue
                    if 'text_contains' in filters and filters['text_contains'] not in chunk_metadata['text'].lower():
                        continue
                
                retrieved_chunks_data.append({
                    'text': chunk_metadata['text'],
                    'url': chunk_metadata['url'],
                    'title': chunk_metadata['title'],
                    'original_index': idx,
                    'distance': distances[0][i],  
                    'info_density': calculate_info_density(chunk_metadata['text'])
                })
        
        return retrieved_chunks_data[:k_value]
    
    except Exception as e:
        print(f"Error during RAG retrieval for query '{query_text}': {str(e)}")
        return []

def calculate_info_density(text):
    """
    Calculate information density based on text features.
    Higher values indicate more informative content.
    """
    # Count content-indicating features
    features = {
        'has_list': 1 if any(marker in text for marker in ['• ', '* ', '- ', '\n- ', '\n* ', '\n• ']) else 0,
        'has_numeric': len(re.findall(r'\d+', text)) / max(1, len(text.split())),
        'has_headings': 1 if re.search(r'#{1,6} ', text) else 0,
        'text_length': min(1.0, len(text) / 300),  
        'keyword_ratio': sum(1 for word in ['feature', 'benefit', 'size', 'color', 'material', 
                                           'instruction', 'detail', 'specification', 'warning'] 
                             if word in text.lower()) / 9
    }
    
    # Avoid excessively short chunks that are often just titles
    if len(text.strip()) < 50:
        return 0.1  
    
    # Penalize chunks that are just product headers with no actual info
    if len(text.strip().split()) < 10 and ("Product:" in text or text.count('#') > 0):
        return 0.2
    
    # Calculate weighted density (weights could be tuned based on empirical testing)
    density = (
        0.2 * features['has_list'] + 
        0.2 * features['has_numeric'] + 
        0.1 * features['has_headings'] + 
        0.3 * features['text_length'] +
        0.2 * features['keyword_ratio']
    )
    
    return density

def re_rank_chunks(chunks, query, intent_data=None, max_chunks=6):
    """
    Re-rank retrieved chunks based on query relevance and information density.
    
    Args:
        chunks: List of chunk data from perform_rag_retrieval
        query: Original query or enhanced query
        intent_data: Query intent analysis results
        max_chunks: Maximum chunks to return
    
    Returns:
        List of re-ranked chunks limited to max_chunks
    """
    if not chunks:
        return []
    
    # Extract all chunk texts for TF-IDF processing
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Create TF-IDF vectorizer for better keyword matching
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(chunk_texts + [query])
        # Get similarity between query (last document) and each chunk
        query_vector = tfidf_matrix[-1]
        chunk_vectors = tfidf_matrix[:-1]
        tfidf_similarities = chunk_vectors.dot(query_vector.T).toarray().flatten()
    except:
        # Fallback if TF-IDF fails
        tfidf_similarities = [0.5] * len(chunks)  
    
    # Prepare keywords based on intent
    intent_keywords = []
    if intent_data:
        if intent_data.get('intent_type') == 'product_specific':
            info_need = intent_data.get('information_need', '')
            if info_need == 'usage':
                intent_keywords = ['use', 'instruction', 'step', 'how to', 'guide', 'apply']
            elif info_need == 'benefits':
                intent_keywords = ['benefit', 'advantage', 'help', 'improve', 'effect']
            elif info_need == 'materials':
                intent_keywords = ['material', 'made of', 'composition', 'fabric', 'construction']
            elif info_need == 'features':
                intent_keywords = ['feature', 'function', 'capability', 'include', 'provide']
            elif info_need == 'warnings':
                intent_keywords = ['warning', 'caution', 'avoid', 'risk', 'danger', 'side effect']
            elif info_need == 'sizing':
                intent_keywords = ['size', 'measurement', 'dimension', 'fit', 'small', 'medium', 'large']
            elif info_need == 'conditions':
                intent_keywords = ['condition', 'treat', 'therapy', 'disorder', 'symptom', 'diagnose']
            elif info_need == 'age_range':
                intent_keywords = ['age', 'child', 'adult', 'toddler', 'infant', 'teen', 'appropriate']
    
    for i, chunk in enumerate(chunks):
        # Calculate combined score based on multiple factors
        # 1. Vector similarity from RAG retrieval (already normalized as distance)
        rag_score = max(0, 1.0 - chunk['distance']) 
        
        # 2. TF-IDF similarity 
        tfidf_score = tfidf_similarities[i]
        
        # 3. Information density score (from calculate_info_density)
        density_score = chunk['info_density']
        
        # 4. Intent-specific keyword matching
        keyword_score = 0
        if intent_keywords:
            matches = sum(1 for keyword in intent_keywords if keyword in chunk['text'].lower())
            keyword_score = min(1.0, matches / max(1, len(intent_keywords)))
        
        # 5. Special handling for very short or header-only chunks (often uninformative)
        text_length = len(chunk['text'].strip())
        if text_length < 50:
            length_penalty = 0.5  
        elif text_length < 100:
            length_penalty = 0.8  
        else:
            length_penalty = 1.0  
        
        # 6. Calculate final combined score
        # Weight factors - these can be tuned based on empirical testing
        chunk['final_score'] = (
            0.25 * rag_score +      
            0.25 * tfidf_score +   
            0.2 * density_score +  
            0.2 * keyword_score +   
            0.1                     
        ) * length_penalty          
        
        # Debug info
        print(f"Chunk {i} (len={text_length}): RAG={rag_score:.2f}, TF-IDF={tfidf_score:.2f}, "
              f"Density={density_score:.2f}, Keywords={keyword_score:.2f}, "
              f"Final={chunk['final_score']:.2f}")
    
    # Sort by combined score (descending)
    ranked_chunks = sorted(chunks, key=lambda x: x['final_score'], reverse=True)
    
    # De-duplicate content if very similar chunks exist
    unique_chunks = []
    seen_content = set()
    for chunk in ranked_chunks:
        # Create a simplified fingerprint of the content to detect near-duplicates
        fingerprint = ' '.join(chunk['text'].lower().split()[:20])
        if fingerprint not in seen_content:
            unique_chunks.append(chunk)
            seen_content.add(fingerprint)
            if len(unique_chunks) >= max_chunks:
                break
    
    return unique_chunks[:max_chunks]

def generate_targeted_queries(intent_data):
    """
    Generate targeted retrieval queries based on query intent analysis.
    
    Args:
        intent_data: Query intent analysis results
    
    Returns:
        List of (query_text, importance_weight) tuples
    """
    queries = []
    
    if intent_data['intent_type'] == 'product_general':
        product = intent_data['product']
        # Multiple queries with different emphases for general product info
        queries = [
            (f"Detailed product information, features and overview for {product}", 1.0),
            (f"Technical specifications and materials for {product}", 0.8),
            (f"Instructions and usage guidelines for {product}", 0.8),
            (f"Benefits and advantages of using {product}", 0.7)
        ]
    
    elif intent_data['intent_type'] == 'product_specific':
        product = intent_data['product']
        info_need = intent_data['information_need']
        
        # Primary query focused on the specific information need
        if info_need == 'usage':
            queries = [
                (f"Instructions and steps for how to use {product}", 1.0),
                (f"Usage guidelines and application instructions for {product}", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'benefits':
            queries = [
                (f"Benefits and advantages of using {product}", 1.0),
                (f"How {product} helps users and its positive effects", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'materials':
            queries = [
                (f"Material composition and construction of {product}", 1.0),
                (f"What {product} is made of and its components", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'features':
            queries = [
                (f"Features and capabilities of {product}", 1.0),
                (f"What {product} does and its functionality", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'warnings':
            queries = [
                (f"Warnings, precautions and contraindications for {product}", 1.0),
                (f"Safety information and risks associated with {product}", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'sizing':
            queries = [
                (f"Sizing information and fitting guide for {product}", 1.0),
                (f"Size chart and measurement details for {product}", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'conditions':
            queries = [
                (f"Medical conditions and issues that {product} helps with", 1.0),
                (f"Therapeutic uses and applications of {product}", 0.9),
                (f"General information about {product}", 0.5)
            ]
        elif info_need == 'age_range':
            queries = [
                (f"Age range and appropriate ages for using {product}", 1.0),
                (f"Whether {product} is for children, adults, or specific age groups", 0.9),
                (f"General information about {product}", 0.5)
            ]
        else:
            # Default queries for other specific intents
            queries = [
                (f"Information about {info_need} for {product}", 1.0),
                (f"General details and specifications for {product}", 0.7)
            ]
    
    elif intent_data['intent_type'] == 'comparison':
        products = intent_data['products']
        aspects = intent_data['aspects']
        
        # Generate queries for each product individually
        for product in products:
            if aspects == ["general"]:
                queries.append((f"Detailed features and specifications for {product}", 0.9))
            else:
                for aspect in aspects:
                    queries.append((f"{aspect} information for {product}", 0.9))
        
        # Also try direct comparison queries
        if len(products) == 2:
            if aspects == ["general"]:
                queries.append((f"Comparison between {products[0]} and {products[1]}", 1.0))
            else:
                for aspect in aspects:
                    queries.append((f"Compare {aspect} of {products[0]} and {products[1]}", 1.0))
    
    elif intent_data['intent_type'] == 'category_listing':
        category = intent_data['category']
        queries = [
            (f"List of products in the {category} category", 1.0),
            (f"Products classified as {category}", 0.9),
            (f"{category} product collection and options", 0.8)
        ]
    
    else:  
        # Use the original query with a slight reformulation
        original_query = intent_data['query']
        queries = [
            (original_query, 1.0),
            (f"Information about {original_query}", 0.8)
        ]
    
    return queries

def get_product_slug(product_name):
    """Generate a URL-friendly slug from a product name"""
    if not product_name:
        return ""
    
    slug = product_name.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('®', '')
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    
    return slug

def multi_query_retrieval(intent_data, max_chunks_per_query=5, max_total_chunks=15):
    """
    Perform retrieval using multiple targeted queries based on intent.
    
    Args:
        intent_data: Query intent analysis results
        max_chunks_per_query: Max chunks to retrieve per query
        max_total_chunks: Max total chunks to retrieve across all queries
    
    Returns:
        List of retrieved chunks
    """
    # Generate targeted queries
    targeted_queries = generate_targeted_queries(intent_data)
    
    all_chunks = []
    seen_indices = set() 
    
    # If we're looking for a specific product, create URL filters
    url_filters = {}
    if intent_data['intent_type'] in ['product_general', 'product_specific']:
        product = intent_data['product']
        slug = get_product_slug(product)
        if slug:
            url_filters = {'url_contains': f'/products/{slug}'}
    
    # Special handling for comparison intent
    if intent_data['intent_type'] == 'comparison':
        products = intent_data['products']
        for product in products:
            # Set a per-product chunk budget
            per_product_chunks = max_total_chunks // len(products)
            
            # Create product-specific filters
            product_slug = get_product_slug(product)
            if product_slug:
                product_filters = {'url_contains': f'/products/{product_slug}'}
            else:
                product_filters = {'text_contains': product.lower()}
            
            # Retrieve chunks for specific aspects of this product
            aspects = intent_data['aspects']
            if aspects == ["general"]:
                query = f"Detailed information about {product}"
                chunks = perform_rag_retrieval(query, per_product_chunks, filters=product_filters)
                
                for chunk in chunks:
                    if chunk['original_index'] not in seen_indices:
                        chunk['product'] = product  
                        all_chunks.append(chunk)
                        seen_indices.add(chunk['original_index'])
            else:
                # Divide chunks among aspects
                per_aspect_chunks = max(1, per_product_chunks // len(aspects))
                for aspect in aspects:
                    query = f"{aspect} information for {product}"
                    chunks = perform_rag_retrieval(query, per_aspect_chunks, filters=product_filters)
                    
                    for chunk in chunks:
                        if chunk['original_index'] not in seen_indices:
                            chunk['product'] = product
                            chunk['aspect'] = aspect
                            all_chunks.append(chunk)
                            seen_indices.add(chunk['original_index'])
    else:
        # For non-comparison intents, execute each targeted query
        for query_text, importance in targeted_queries:
            print(f"Executing targeted query: '{query_text}' (importance: {importance})")
            
            # Adjust chunk count based on query importance
            chunks_to_retrieve = max(1, int(max_chunks_per_query * importance))
            
            # Perform retrieval with appropriate filters
            retrieved_chunks = perform_rag_retrieval(query_text, chunks_to_retrieve, filters=url_filters)
            
            # Add unique chunks to the result set
            for chunk in retrieved_chunks:
                if chunk['original_index'] not in seen_indices and len(all_chunks) < max_total_chunks:
                    chunk['query_source'] = query_text 
                    chunk['query_importance'] = importance
                    all_chunks.append(chunk)
                    seen_indices.add(chunk['original_index'])
    
    return all_chunks

def select_final_chunks(all_retrieved_chunks, intent_data, user_message, max_final_chunks=6):
    """
    Select the final set of chunks to include in the LLM context.
    
    Args:
        all_retrieved_chunks: Combined chunks from multi-query retrieval
        intent_data: Query intent analysis results
        user_message: Original user message
        max_final_chunks: Maximum chunks to return
    
    Returns:
        List of selected chunks for LLM context
    """
    # Special handling for comparison intent
    if intent_data['intent_type'] == 'comparison':
        products = intent_data['products']
        aspects = intent_data['aspects']
        
        # Group chunks by product
        product_chunks = defaultdict(list)
        for chunk in all_retrieved_chunks:
            product = chunk.get('product', 'unknown')
            product_chunks[product].append(chunk)
        
        final_chunks = []
        # Allocate chunks per product, ensuring equal representation
        chunks_per_product = max(1, max_final_chunks // len(products))
        
        for product in products:
            # Select best chunks for this product
            product_specific_chunks = product_chunks.get(product, [])
            if product_specific_chunks:
                # Re-rank the product's chunks specifically for this query
                ranked_product_chunks = re_rank_chunks(
                    product_specific_chunks, 
                    user_message,
                    intent_data,
                    max_chunks=chunks_per_product
                )
                final_chunks.extend(ranked_product_chunks)
        
        # If we still have room after allocating per product, add more from the highest scoring chunks
        if len(final_chunks) < max_final_chunks:
            # Get remaining chunks not already selected
            remaining_indices = set(chunk['original_index'] for chunk in final_chunks)
            remaining_chunks = [c for c in all_retrieved_chunks if c['original_index'] not in remaining_indices]
            
            # Re-rank remaining chunks
            if remaining_chunks:
                additional_chunks = re_rank_chunks(
                    remaining_chunks,
                    user_message,
                    intent_data,
                    max_chunks=(max_final_chunks - len(final_chunks))
                )
                final_chunks.extend(additional_chunks)
        
        return final_chunks[:max_final_chunks]
    
    # For non-comparison intents, simply re-rank all retrieved chunks
    return re_rank_chunks(all_retrieved_chunks, user_message, intent_data, max_chunks=max_final_chunks)

@require_http_methods(["POST"])
def process_message(request):
    """Process user message and generate a response using the RAG pipeline."""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({'error': 'Empty message'}, status=400)
        
        # Initialize or retrieve chat history
        conversation_id = data.get('conversation_id', '')
        chat_history = data.get('history', [])
        
        # 1. Extract potential product name from user message
        potential_product = extract_potential_product_name(user_message)
        print(f"Extracted potential product: {potential_product}")
        
        # 2. Analyze query intent
        intent_data = analyze_query_intent(user_message, potential_product)
        print(f"Query intent analysis: {intent_data}")
        
        # 3. Perform multi-query retrieval based on intent
        all_retrieved_chunks = multi_query_retrieval(
            intent_data, 
            max_chunks_per_query=5, 
            max_total_chunks=15
        )
        
        # 4. Select final chunks for LLM context
        selected_chunks = select_final_chunks(
            all_retrieved_chunks, 
            intent_data, 
            user_message, 
            max_final_chunks=6
        )
        
        # 5. Format chunks for LLM context
        context_text = format_chunks_for_context(selected_chunks, intent_data)
        
        # 6. Generate response using Gemini
        response_text = generate_llm_response(user_message, context_text, chat_history, intent_data)
        
        # 7. Log the interaction
        if conversation_id:
            ChatLog.objects.create(
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_response=response_text,
                query_intent=json.dumps(intent_data),
                chunks_used=json.dumps([{
                    'text': chunk['text'][:100] + '...', 
                    'url': chunk['url'],
                    'title': chunk['title'],
                    'score': chunk.get('final_score', 0)
                } for chunk in selected_chunks])
            )
        
        return JsonResponse({
            'response': response_text,
            'conversation_id': conversation_id,
            'product_detected': potential_product,
            'intent': intent_data['intent_type']
        })
    
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

def format_chunks_for_context(chunks, intent_data):
    """
    Format retrieved chunks into context for the LLM.
    Adds special tags and organizes information based on intent.
    
    Args:
        chunks: List of selected chunks
        intent_data: Query intent analysis results
    
    Returns:
        Formatted context text for the LLM
    """
    context_parts = []
    
    # Special handling for comparison intent
    if intent_data['intent_type'] == 'comparison':
        products = intent_data['products']
        products_chunks = {p: [] for p in products}
        
        # Group chunks by product
        for chunk in chunks:
            product = chunk.get('product')
            if product in products_chunks:
                products_chunks[product].append(chunk)
        
        # Format chunks for each product
        for product in products:
            if products_chunks[product]:
                context_parts.append(f"\n[PRODUCT INFORMATION: {product}]")
                for chunk in products_chunks[product]:
                    # Add aspect tag if available
                    aspect_tag = f" - {chunk['aspect'].upper()}" if 'aspect' in chunk else ""
                    context_parts.append(f"[Direct Info for {product}{aspect_tag}] {chunk['text']}")
                    context_parts.append(f"Source: {chunk['url']} | {chunk['title']}")
    else:
        # For non-comparison intents, format each chunk
        for chunk in chunks:
            # Determine if this is direct product info
            is_direct_info = False
            tag_prefix = "[Retrieved Info]"
            
            if intent_data['intent_type'] in ['product_general', 'product_specific']:
                product = intent_data['product']
                if product and product.lower() in chunk['text'].lower():
                    tag_prefix = f"[Direct Info for {product}]"
                    is_direct_info = True
            
            # Add formatted chunk to context
            context_parts.append(f"{tag_prefix} {chunk['text']}")
            context_parts.append(f"Source: {chunk['url']} | {chunk['title']}")
    
    return "\n\n".join(context_parts)

def generate_llm_response(user_message, context_text, chat_history, intent_data):
    """
    Generate a response using the LLM (Gemini) based on the user message,
    retrieved context, and chat history.
    
    Args:
        user_message: Original user message
        context_text: Formatted context from retrieved chunks
        chat_history: Previous conversation history
        intent_data: Query intent analysis results
    
    Returns:
        Generated response text
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Construct system prompt based on intent
        system_prompt = get_system_prompt(intent_data)
        
        # Format the context and query for the LLM
        complete_prompt = f"{system_prompt}\n\n[RETRIEVED CONTEXT]\n{context_text}\n\n[USER QUERY]\n{user_message}\n\n[YOUR RESPONSE]"
        
        # Generate the response
        response = model.generate_content(complete_prompt)
        response_text = response.text
        
        # Post-process the response if needed
        response_text = post_process_response(response_text, intent_data)
        
        return response_text
        
    except Exception as e:
        print(f"Error generating LLM response: {str(e)}")
        return "I'm sorry, I encountered an error while trying to answer your question. Please try again or rephrase your question."

def get_system_prompt(intent_data):
    """
    Generate an appropriate system prompt based on the query intent.
    
    Args:
        intent_data: Query intent analysis results
    
    Returns:
        System prompt for the LLM
    """
    base_prompt = """You are an informative assistant for Sensory Souk, helping employees find information about products. 
    
Your task is to provide accurate answers based ONLY on the retrieved context I'll provide. Follow these guidelines:

1. Use ONLY the information in the [RETRIEVED CONTEXT] section to answer the user's query.
2. If the retrieved context doesn't contain sufficient information to fully answer the query, clearly say "Based on the available information, I cannot provide details about [specific missing information]."
3. Prioritize information from sections marked as [Direct Info] as they are most relevant.
4. Synthesize information from multiple chunks when needed to provide a comprehensive answer.
5. Present information in a clear, organized manner appropriate to the query type.
6. Do NOT make up or infer information that isn't explicitly mentioned in the retrieved context.
7. When information appears to be incomplete, acknowledge the limitation rather than guessing.
8. Cite the sources of your information by mentioning the product or document name."""

    # Add intent-specific instructions
    if intent_data['intent_type'] == 'product_specific':
        info_need = intent_data.get('information_need', '')
        product = intent_data.get('product', '')
        
        base_prompt += f"\n\nThis query is specifically about the {info_need} of {product}. Focus your answer on this aspect while providing a complete response."
        
        if info_need == 'usage':
            base_prompt += "\nIf usage instructions are available, present them as a numbered list for clarity."
        elif info_need == 'benefits':
            base_prompt += "\nOrganize benefits in a clear manner, highlighting key advantages."
        elif info_need == 'materials':
            base_prompt += "\nClearly list all materials mentioned in the context."
        elif info_need == 'warnings':
            base_prompt += "\nEnsure all warnings and precautions are prominently included in your response."
    
    elif intent_data['intent_type'] == 'comparison':
        products = intent_data.get('products', [])
        aspects = intent_data.get('aspects', [])
        
        products_str = " and ".join(products)
        aspects_str = ", ".join(aspects) if aspects != ["general"] else "all aspects"
        
        base_prompt += f"\n\nThis query asks for a comparison between {products_str} regarding {aspects_str}. Structure your response to clearly compare these products side by side on relevant attributes. If information for one product is missing on certain aspects, acknowledge this limitation."
    
    elif intent_data['intent_type'] == 'category_listing':
        category = intent_data.get('category', '')
        base_prompt += f"\n\nThis query asks for a listing of products in the {category} category. Present the products as a list if multiple are found."
    
    return base_prompt

def post_process_response(response_text, intent_data):
    """
    Apply any necessary post-processing to the LLM's response.
    
    Args:
        response_text: The raw response from the LLM
        intent_data: Query intent analysis results
    
    Returns:
        Post-processed response text
    """
    # Remove any "assistant:" or similar prefixes if present
    response_text = re.sub(r'^(Assistant:|AI:|Response:)\s*', '', response_text, flags=re.IGNORECASE)
    
    # Ensure response doesn't end with an incomplete sentence
    if response_text and not re.search(r'[.!?]\s*$', response_text):
        response_text = response_text.rstrip() + "."
    
    # For product listings, ensure consistent formatting
    if intent_data['intent_type'] == 'category_listing':
        # If response contains a list but isn't formatted properly
        if ":" in response_text and not re.search(r'^\s*[-*•]', response_text, re.MULTILINE):
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if ':' in line and i > 0:
                    product_name = line.split(':')[0].strip()
                    lines[i] = f"• {product_name}"
            response_text = '\n'.join(lines)
    
    return response_text

@require_http_methods(["GET"])
def get_chat_history(request):
    """
    Retrieve chat history for a given conversation ID.
    Used by the frontend to load previous conversations.
    """
    conversation_id = request.GET.get('conversation_id', '')
    
    if not conversation_id:
        return JsonResponse({'error': 'No conversation ID provided'}, status=400)
    
    try:
        # Query chat logs for this conversation
        chat_logs = ChatLog.objects.filter(conversation_id=conversation_id).order_by('created_at')
        
        # Format history for frontend
        history = []
        for log in chat_logs:
            history.append({
                'user': log.user_message,
                'assistant': log.assistant_response,
                'timestamp': log.created_at.isoformat()
            })
        
        return JsonResponse({'history': history})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["DELETE"])
def clear_chat_history(request):
    """
    Clear chat history for a given conversation ID.
    Used by the frontend to start fresh conversations.
    """
    try:
        data = json.loads(request.body)
        conversation_id = data.get('conversation_id', '')
        
        if not conversation_id:
            return JsonResponse({'error': 'No conversation ID provided'}, status=400)
        
        # Delete chat logs for this conversation
        ChatLog.objects.filter(conversation_id=conversation_id).delete()
        
        return JsonResponse({'success': True, 'message': 'Chat history cleared'})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)