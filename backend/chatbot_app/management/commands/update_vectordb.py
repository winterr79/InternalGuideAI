#!/usr/bin/env python
import os
import json
import time
import logging
import hashlib
# import pickle
import numpy as np
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import faiss
from tqdm import tqdm
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
# import sys
# sys.setrecursionlimit(10000)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Google Generative AI for embeddings
try:
    import google.generativeai as genai
except ImportError:
    logger.error("The google-generativeai package is required. Install it with: pip install google-generativeai")
    raise

class WebCrawler:
    """
    Handles crawling websites, extracting content, and managing the crawl state.
    """
    
    def __init__(self, base_url=None, max_pages=100):
        """
        Initialize the web crawler.
        
        Args:
            base_url (str): Base URL to start crawling from
            max_pages (int): Maximum number of pages to crawl
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = []
        self.content_cache = {}
        
        # Load existing cache if available
        self.cache_path = os.path.join(settings.BASE_DIR, 'data', 'content_cache.json')
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self.load_cache()
        
    def load_cache(self):
        """Load existing content cache if available."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    self.content_cache = json.load(f)
                logger.info(f"Loaded content cache with {len(self.content_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {e}. Starting fresh.")
            self.content_cache = {}
            
    def save_cache(self):
        """Save content cache to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.content_cache, f)
            logger.info(f"Saved content cache with {len(self.content_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def crawl(self, start_url=None, force=False):
        """
        Start crawling from a given URL.
        
        Args:
            start_url (str): URL to start crawling from. If None, uses self.base_url
            force (bool): Whether to force recrawling of pages in cache
            
        Returns:
            dict: Dictionary mapping URLs to extracted content
        """
        if start_url is None:
            if self.base_url is None:
                raise ValueError("No start_url or base_url provided for crawling")
            start_url = self.base_url
            
        base_domain = urlparse(start_url).netloc
        self.visited = set()
        self.to_visit = [start_url]
        
        pages_content = {}
        crawled_count = 0
        
        while self.to_visit and crawled_count < self.max_pages:
            url = self.to_visit.pop(0)
            
            if url in self.visited:
                continue
                
            self.visited.add(url)
            crawled_count += 1
            
            logger.info(f"Crawling ({crawled_count}/{self.max_pages}): {url}")
            
            try:
                # Check cache first
                content_hash = None
                should_process = True
                
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status_code}")
                    continue
                    
                html_content = response.text
                content_hash = hashlib.md5(html_content.encode()).hexdigest()
                
                # Check if content has changed
                if url in self.content_cache and self.content_cache[url]['hash'] == content_hash and not force:
                    logger.info(f"Content unchanged, using cached data for: {url}")
                    should_process = False
                    if 'extracted_text' in self.content_cache[url]:
                        pages_content[url] = {
                            'title': self.content_cache[url].get('title', ''),
                            'text': self.content_cache[url]['extracted_text'],
                            'url': url
                        }
                
                if should_process:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract title and text content
                    title = soup.title.string if soup.title else url
                    content = self._extract_text_content(soup)
                    
                    if not content.strip():
                        logger.warning(f"No content extracted from {url}")
                        continue
                    
                    # Store extracted content
                    pages_content[url] = {
                        'title': title,
                        'text': content,
                        'url': url
                    }
                    
                    # Update cache
                    self.content_cache[url] = {
                        'hash': content_hash,
                        'last_crawled': datetime.now().isoformat(),
                        'title': title,
                        'extracted_text': content
                    }
                    
                    # Find links to other pages on the same domain
                    if crawled_count < self.max_pages:
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(url, href)
                            parsed_full_url = urlparse(full_url)
                            url_to_consider = parsed_full_url._replace(fragment="").geturl()
                            if parsed_full_url.netloc == base_domain:
                                if url_to_consider not in self.visited and url_to_consider not in self.to_visit:
                                    path = parsed_full_url.path.lower()
                                    is_relevant_path = False
                                    if path.startswith('/products/') or path.startswith('/product/'):
                                        is_relevant_path = True
                                    if path.startswith('/collections/') or path.startswith('/categories/') or path.startswith('/category/'):
                                        is_relevant_path = True
                                    irrelevant_terms = ['cart', 'account', 'policy', 'search', 'checkout', 'login', 'register', 'password']
                                    if any(skip_term in path for skip_term in irrelevant_terms):
                                        is_relevant_path = False
                                    # if len(path.strip('/')) < 5 and not is_relevant_path:
                                    #     is_relevant_path = False
                                    if any(path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip']):
                                        is_relevant_path = False
                                    if is_relevant_path:
                                        if len(self.to_visit) < (self.max_pages - crawled_count):
                                            logger.debug(f"Queueing relevant URL: {url_to_consider}")
                                            self.to_visit.append(url_to_consider)
                                        else:
                                            logger.debug(f"Max pages limit reached, not queueing: {url_to_consider}")
                                    else:
                                        logger.debug(f"Skipping irrelevant URL: {url_to_consider} (Path: {path})")

                            
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                
        # Save updated cache
        self.save_cache()
        
        return pages_content
        
    # Inside WebCrawler class

    # Inside WebCrawler class
    def _extract_text_content(self, soup):
        if soup is None: # Should not happen if requests.get was successful
            logger.error("Soup object is None in _extract_text_content.")
            return ""

        # 1. Remove global non-content areas first
        for element_type in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
            for element in soup.find_all(element_type):
                if element: # Check if element is not None before decomposing
                    element.decompose()

        target_soup_for_extraction = None
        logger.debug("Attempting to find specific product description section...")

        # 2. Try to find the specific "Product Description" heading
        product_desc_heading = soup.find('h2', string=lambda text: text and "product description" in text.lower())
        
        if product_desc_heading:
            logger.debug(f"Found heading: '{product_desc_heading.get_text(strip=True)}'")
            heading_parent_div = product_desc_heading.find_parent('div', class_='ungroup-description-tab__heading')
            if heading_parent_div:
                content_div = heading_parent_div.find_next_sibling('div', class_='mt0 mt--first-child-0 body2')
                if content_div:
                    logger.debug("Found specific content div (mt0 mt--first-child-0 body2) next to product description heading.")
                    target_soup_for_extraction = content_div
                else:
                    logger.debug("Could not find 'mt0 mt--first-child-0 body2' div next to product description heading's parent.")
            else:
                logger.debug("Could not find parent 'ungroup-description-tab__heading' for product description heading.")
        else:
            logger.debug("Did not find 'Product Description' h2 heading using specific string search. This is expected for category pages.")

        # 3. Fallback to other potential selectors (more for product pages)
        if not target_soup_for_extraction:
            logger.debug("Specific heading-based search failed or not applicable, trying general product selectors...")
            potential_selectors = [
                {'name': 'div', 'attrs': {'class': 'product-single__description'}},
                {'name': 'div', 'attrs': {'class': 'product__description'}},
                {'name': 'div', 'attrs': {'class': 'rte'}},
                {'name': 'article', 'attrs': {'class': 'product-page-content'}},
            ]
            for selector_info in potential_selectors:
                found_area = soup.find(selector_info['name'], **selector_info['attrs'])
                if found_area:
                    logger.debug(f"Found specific product description area using general selector: {selector_info}")
                    target_soup_for_extraction = found_area
                    break
        
        # 4. Fallback to broader main content (this will likely be hit for CATEGORY pages)
        if not target_soup_for_extraction:
            logger.debug("General product selectors failed, falling back to main content area logic (expected for category pages).")
            main_content_area = soup.find('main') or \
                                soup.find(id='content') or \
                                soup.find(class_=['content', 'main-content', 'shopify-section', 'product-template', 'template-collection']) # Added template-collection
            if main_content_area:
                logger.debug(f"Using broader main content area: {main_content_area.name} {main_content_area.get('class', '')} {main_content_area.get('id', '')}")
                target_soup_for_extraction = main_content_area
                # Clean common noise from broader main content
                noisy_classes_to_remove = ['related-products', 'product-form', 'social-sharing', 'reviews', 
                                           'product-gallery', 'breadcrumb', 'toolbar', 'sorter', 'filter', 
                                           'pagination', 'facets-container', 'site-footer'] # Added more
                for element in list(target_soup_for_extraction.find_all(True, class_=lambda c: c and any(x in str(c).lower() for x in noisy_classes_to_remove))): # Iterate over a copy
                    if element and element.parent: # Ensure element exists and has a parent
                        logger.debug(f"Decomposing noisy element in main content: <{element.name} class='{element.get('class')}'>")
                        element.decompose()
            else:
                logger.warning("No specific or main content area found. Falling back to whole soup (might be noisy).")
                target_soup_for_extraction = soup

        # 5. Extract text from the determined target_soup_for_extraction
        content_parts = []
        if target_soup_for_extraction: # Ensure target_soup_for_extraction is not None
            # Extract H2, H3, H4 as potential subheadings or titles within the content
            for heading in target_soup_for_extraction.find_all(['h2', 'h3', 'h4']):
                prefix = {'h2': '## ', 'h3': '### ', 'h4': '#### '}.get(heading.name, '')
                text = heading.get_text(strip=True)
                if text:
                    content_parts.append(f"{prefix}{text}")

            # Extract paragraphs and list items
            for p_or_li in target_soup_for_extraction.find_all(['p', 'li']):
                text = p_or_li.get_text(strip=True)
                noise_phrases = [
                    "add to cart", "decrease quantity", "increase quantity", "open media", 
                    "view full details", "vendor:", "sold out", "buy it now", "ask a question", 
                    "share", "regular price", "sale price", "tax included", 
                    "shipping calculated at checkout", "unit price", "/ per", "you may also like", 
                    "recently viewed", "customer reviews", "write a review", "availability:", 
                    "in stock", "quantity", "default title", "select options", "sort by", 
                    "filter by", "view as:", "items per page", "quick view" # Added more
                ]
                if text and len(text) > 10 and not any(phrase in text.lower() for phrase in noise_phrases):
                    text = re.sub(r'^\s*[\d\.,]+\s*qar\s*-\s*', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'\s*qar\s*$', '', text, flags=re.IGNORECASE)
                    if len(text.split()) > 2:
                        content_parts.append(text)
            
            # For category pages, you might also want to grab product titles if they are not linked
            # This is a heuristic and might need adjustment
            # Example: Look for product titles in common grid item structures
            for item_title_element in target_soup_for_extraction.find_all(class_=['product-card__title', 'product-item-meta__title', 'grid-view-item__title']):
                title_text = item_title_element.get_text(strip=True)
                if title_text and title_text not in content_parts and len(title_text.split()) > 1: # Avoid single words
                    logger.debug(f"Found potential product title on category page: {title_text}")
                    content_parts.append(f"Product listed: {title_text}")

        else:
            logger.error("Critical: target_soup_for_extraction was None before text parsing. This should not happen.")
            return "" # Return empty string if target is None

        full_text = "\n\n".join(content_parts)
        full_text = re.sub(r'\n\s*\n', '\n\n', full_text).strip()

        if not full_text:
            logger.warning(f"Extraction resulted in empty text for URL (after filtering).")
            
        return full_text


class TextProcessor:
    """
    Handles text processing operations like chunking and embedding generation.
    """
    
    def __init__(self, api_key=None):
        """Initialize the text processor with necessary components."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("No Google API key provided. Embedding generation will not work.")
    
    def chunk_text(self, text, chunk_size=300, chunk_overlap=50):
        """
        Enhanced chunking strategy that preserves semantic units better.
    
        This function attempts to:
        1. Keep headings with their content
        2. Respect paragraph boundaries where possible
        3. Handle lists appropriately
        4. Maintain semantic coherence with sufficient overlap
    
        Args:
            text (str): The text to chunk
            chunk_size (int): Target size of each chunk
            chunk_overlap (int): Number of overlapping tokens between chunks
    
        Returns:
            list: List of text chunks
        """
        # Split text into semantic sections (preserve headings with content)
        heading_pattern = r'(^|\n)(#+\s+.+)(\n|$)'
        sections = []
        last_end = 0
    
        # Find all headings and use them as section boundaries
        for match in re.finditer(heading_pattern, text, re.MULTILINE):
            heading_start = match.start()
            if heading_start > last_end:
                # Add content between last heading and this one
                sections.append(text[last_end:heading_start])
            
            # Find where this section ends (at the next heading or end of text)
            next_match = re.search(heading_pattern, text[match.end():], re.MULTILINE)
            if next_match:
                section_end = match.end() + next_match.start()
            else:
                section_end = len(text)
            
            # Add this headed section
            sections.append(text[match.start():section_end])
            last_end = section_end
    
        # Add final section if needed
        if last_end < len(text):
            sections.append(text[last_end:])
    
        # If no sections were found (no headings), use paragraph splitting
        if not sections or (len(sections) == 1 and sections[0] == text):
            # Split by paragraphs (double line breaks)
            sections = [s for s in re.split(r'\n\s*\n', text) if s.strip()]
    
        # Further chunk large sections
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # For large sections, attempt to break at paragraph or sentence boundaries
                paragraphs = [p for p in re.split(r'\n\s*\n', section) if p.strip()]
            
                current_chunk = ""
                for paragraph in paragraphs:
                    # If paragraph itself exceeds chunk size, break it into sentences
                    if len(paragraph) > chunk_size:
                        sentences = [s.strip() + "." for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
                    
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                                current_chunk += "\n" + sentence if current_chunk else sentence
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                            
                                # If single sentence exceeds chunk size, we have to break it
                                if len(sentence) > chunk_size:
                                    sentence_chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size-chunk_overlap)]
                                    chunks.extend(sentence_chunks)
                                    current_chunk = sentence_chunks[-1][-chunk_overlap:] if sentence_chunks else ""
                                else:
                                    current_chunk = sentence
                    else:
                        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                        else:
                            chunks.append(current_chunk)
                            current_chunk = paragraph
            
                if current_chunk:
                    chunks.append(current_chunk)
    
        # Ensure proper overlap between chunks
        result_chunks = []
        prev_chunk_end = ""
    
        for i, chunk in enumerate(chunks):
            # If this isn't the first chunk, prepend overlap from previous chunk
            if i > 0 and prev_chunk_end:
                chunk = prev_chunk_end + "\n" + chunk
            
            # Store end of this chunk for next iteration's overlap
            words = chunk.split()
            if len(words) > chunk_overlap:
                prev_chunk_end = " ".join(words[-chunk_overlap:])
            else:
                prev_chunk_end = chunk
            
            result_chunks.append(chunk.strip())
    
        return result_chunks
        
    def generate_embeddings(self, chunks, model="embedding-001", batch_size=10, max_retries=3):
        """
        Enhanced embedding generation with batching, retry logic, and better error handling.
    
        Args:
            chunks (list): List of text chunks to embed
            model (str): Embedding model name
            batch_size (int): Number of chunks to process in a single API call
            max_retries (int): Maximum number of retries for failed API calls
    
        Returns:
            list: Embeddings for all chunks
        """
        if not self.api_key:
            logger.error("Cannot generate embeddings: No Google API key provided")
            return None
            
        all_embeddings = []
    
        # Process chunks in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i+batch_size]
            batch_embeddings = []
            
            # Try to embed the batch
            for retry in range(max_retries + 1):
                try:
                    # Process one chunk at a time for more reliable results
                    batch_embeddings = []
                    for chunk in batch:
                        qualified_model_name = model if model.startswith("models/") or model.startswith("tunedModels/") else f"models/{model}"
                        response = genai.embed_content(
                            model=qualified_model_name, 
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        batch_embeddings.append(response['embedding'])
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < max_retries:
                        # Implement exponential backoff
                        wait_time = 2 ** retry  # 1, 2, 4 seconds
                        logger.warning(f"Embedding API error: {e}. Retrying in {wait_time}s... ({retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to generate embeddings after {max_retries} retries: {e}")
                        # Generate zero vectors as fallback
                        logger.warning(f"Using zero vectors for chunks {i} to {i+len(batch)-1}")
                        # Get embedding dimensionality from model spec or use a default
                        dim = 768  # Default for embedding-001, adjust if needed
                        batch_embeddings = [np.zeros(dim).tolist() for _ in batch]
            
            all_embeddings.extend(batch_embeddings)
    
        return all_embeddings


# In update_vectordb.py

# ... (other imports)
# import pickle # No longer needed for metadata if using JSONL
# ...

class VectorStore:
    """
    Manages the vector database operations using JSON Lines for metadata.
    """

    def __init__(self, index_path=None, metadata_path=None):
        """
        Initialize the vector store with paths for index and metadata.
        
        Args:
            index_path (str): Path to store the FAISS index
            metadata_path (str): Path to store the metadata (will be .jsonl)
        """
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        self.index_path = index_path or os.path.join(data_dir, 'faiss_index.bin')
        # Change to .jsonl for metadata
        self.metadata_path = metadata_path or os.path.join(data_dir, 'metadata.jsonl')
        self.url_mapping_path = os.path.join(data_dir, 'url_mapping.json')
        
        self.index = None
        self.chunks_metadata = []
        self.url_mapping = {}
        
        self.load_index_and_metadata()
        
    def load_index_and_metadata(self):
        """Load existing FAISS index, metadata (from JSONL), and URL mapping if available."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            
            if os.path.exists(self.metadata_path):
                self.chunks_metadata = [] # Initialize as empty list
                with open(self.metadata_path, 'r', encoding='utf-8') as f: # Read in text mode
                    for line in f:
                        if line.strip(): # Ensure line is not empty
                            try:
                                self.chunks_metadata.append(json.loads(line))
                            except json.JSONDecodeError as je:
                                logger.error(f"Error decoding JSON from metadata line: '{line.strip()}'. Error: {je}")
                logger.info(f"Loaded {len(self.chunks_metadata)} metadata records from {self.metadata_path}")
            
            if os.path.exists(self.url_mapping_path):
                with open(self.url_mapping_path, 'r', encoding='utf-8') as f:
                    self.url_mapping = json.load(f)
                logger.info(f"Loaded URL mapping with {len(self.url_mapping)} entries")
                
            return True
        except Exception as e:
            logger.error(f"Error loading index or metadata: {e}")
            # Reset to empty if loading fails to ensure clean state for new data
            self.index = None
            self.chunks_metadata = []
            self.url_mapping = {}
            return False
            
    def create_or_update_index(self, chunks_text, metadata_list, embeddings):
        """
        Create a new FAISS index or update an existing one.
        Saves metadata as JSON Lines.
        
        Args:
            chunks_text (list): List of text chunks (not directly used for metadata saving anymore, but good for validation)
            metadata_list (list): List of metadata dictionaries
            embeddings (list): List of embeddings
            
        Returns:
            bool: Success status
        """
        try:
            if len(metadata_list) != len(embeddings): # chunks_text might not be needed here if metadata_list contains the text
                logger.error(f"Mismatch in input lengths: metadata={len(metadata_list)}, embeddings={len(embeddings)}")
                return False
                
            embeddings_np = np.array(embeddings).astype('float32')
            
            new_metadata_to_add = []
            new_embeddings_to_add_np = []

            if self.index is None:
                # Create new index
                if not embeddings: # Handle case with no embeddings
                    logger.warning("No embeddings provided to create a new index.")
                    return False
                embedding_dim = len(embeddings[0])
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.chunks_metadata = metadata_list # Store all new metadata
                new_embeddings_to_add_np = embeddings_np
                # Update URL mapping for all new items
                self.url_mapping.update({meta['id']: meta['url'] for meta in metadata_list})
            else:
                # Update existing index: Identify truly new metadata and embeddings
                existing_ids = {meta['id'] for meta in self.chunks_metadata}
                temp_new_embeddings_list = []

                for i, meta in enumerate(metadata_list):
                    if meta['id'] not in existing_ids:
                        self.chunks_metadata.append(meta) # Add to in-memory list
                        new_metadata_to_add.append(meta) # Keep track of what's new for saving if needed (though we save all)
                        temp_new_embeddings_list.append(embeddings_np[i])
                        self.url_mapping[meta['id']] = meta['url'] # Update URL mapping
                
                if temp_new_embeddings_list:
                    new_embeddings_to_add_np = np.array(temp_new_embeddings_list).astype('float32')
                else:
                    new_embeddings_to_add_np = np.array([]) # Empty array

            # Add new embeddings to FAISS index if any
            if new_embeddings_to_add_np.size > 0:
                if new_embeddings_to_add_np.ndim == 1: # Handle single new embedding
                    new_embeddings_to_add_np = new_embeddings_to_add_np.reshape(1, -1)
                self.index.add(new_embeddings_to_add_np)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save all current metadata as JSON Lines (overwrite existing file)
            with open(self.metadata_path, 'w', encoding='utf-8') as f: # Write in text mode
                for meta_item in self.chunks_metadata:
                    json.dump(meta_item, f)
                    f.write('\n')
                
            # Save URL mapping
            with open(self.url_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.url_mapping, f)
                
            logger.info(f"Saved index with {self.index.ntotal} vectors and {len(self.chunks_metadata)} metadata entries (JSONL)")
            return True
            
        except Exception as e:
            logger.error(f"Error creating/updating index: {e}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback
            return False


class Command(BaseCommand):
    """
    Django management command to crawl websites, process content, and update the vector database.
    """
    
    help = 'Update the vector database by crawling websites and processing content'
    
    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            '--url',
            dest='start_url',
            help='URL to start crawling from'
        )
        parser.add_argument(
            '--urls-file',
            dest='urls_file',
            help='File containing URLs to crawl (one per line)'
        )
        parser.add_argument(
            '--max-pages',
            dest='max_pages',
            type=int,
            default=100,
            help='Maximum number of pages to crawl (default: 100)'
        )
        parser.add_argument(
            '--force',
            dest='force',
            action='store_true',
            help='Force recrawling of pages in cache'
        )
        parser.add_argument(
            '--chunk-size',
            dest='chunk_size',
            type=int,
            default=300,
            help='Size of text chunks for embedding (default: 300)'
        )
        parser.add_argument(
            '--chunk-overlap',
            dest='chunk_overlap',
            type=int,
            default=50,
            help='Overlap between text chunks (default: 50)'
        )
        
    def handle(self, *args, **options):
        """
        Main command execution logic.
        """
        # Check for Google API key
        if not settings.GEMINI_API_KEY:
            self.stderr.write(self.style.ERROR("No Google API key found. Set GEMINI_API_KEY in settings."))
            return
            
        # Set up components
        crawler = WebCrawler(max_pages=options['max_pages'])
        text_processor = TextProcessor(api_key=settings.GEMINI_API_KEY)
        vector_store = VectorStore()
        
        # Get URLs to crawl
        urls_to_crawl = []
        
        if options['start_url']:
            urls_to_crawl.append(options['start_url'])
            
        if options['urls_file']:
            try:
                with open(options['urls_file'], 'r') as f:
                    urls_to_crawl.extend([line.strip() for line in f if line.strip()])
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error reading URLs file: {e}"))
                return
                
        if not urls_to_crawl:
            self.stderr.write(self.style.ERROR("No URLs provided. Use --url or --urls-file options."))
            return
            
        self.stdout.write(self.style.SUCCESS(f"Starting crawl of {len(urls_to_crawl)} URLs"))
        
        # Process each URL
        all_chunks = []
        all_metadata = []
        chunk_id_counter = 0
        
        # If we have existing metadata, find the highest ID to avoid collisions
        if vector_store.chunks_metadata:
            try:
                highest_id = max(int(meta['id']) for meta in vector_store.chunks_metadata if 'id' in meta)
                chunk_id_counter = highest_id + 1
            except (ValueError, KeyError):
                chunk_id_counter = len(vector_store.chunks_metadata)
                
        self.stdout.write(f"Starting chunk ID counter at {chunk_id_counter}")
        
        for url in urls_to_crawl:
            self.stdout.write(f"Processing URL: {url}")
            
            # Set base URL for crawler
            crawler.base_url = url
            
            # Crawl the website
            pages_content = crawler.crawl(start_url=url, force=options['force'])
            
            if not pages_content:
                self.stdout.write(self.style.WARNING(f"No content found for {url}"))
                continue
                
            self.stdout.write(f"Found {len(pages_content)} pages for {url}")
            
            # Process each page
            for page_url, page_data in pages_content.items():
                page_text = page_data['text']
                page_title = page_data['title']
                
                # Skip empty pages
                if not page_text or not page_text.strip():
                    continue
                    
                # Create chunks
                page_chunks = text_processor.chunk_text(
                    page_text, 
                    chunk_size=options['chunk_size'], 
                    chunk_overlap=options['chunk_overlap']
                )
                
                if not page_chunks:
                    self.stdout.write(self.style.WARNING(f"No chunks created for {page_url}"))
                    continue
                    
                # Create metadata for each chunk
                for chunk in page_chunks:
                    chunk_metadata = {
                        'id': str(chunk_id_counter),
                        'url': page_url,
                        'title': page_title,
                        'text': chunk,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    all_chunks.append(chunk)
                    all_metadata.append(chunk_metadata)
                    chunk_id_counter += 1
                    
        # Generate embeddings for all chunks
        if all_chunks:
            self.stdout.write(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = text_processor.generate_embeddings(all_chunks)
            
            if embeddings:
                # Update vector store
                success = vector_store.create_or_update_index(all_chunks, all_metadata, embeddings)
                
                if success:
                    self.stdout.write(self.style.SUCCESS(
                        f"Successfully processed {len(urls_to_crawl)} URLs, "
                        f"created {len(all_chunks)} chunks, and updated the vector store."
                    ))
                else:
                    self.stderr.write(self.style.ERROR("Failed to update vector store."))
            else:
                self.stderr.write(self.style.ERROR("Failed to generate embeddings."))
        else:
            self.stdout.write(self.style.WARNING("No chunks created. Vector store not updated."))