import os
import json
import time
import logging
import hashlib
import numpy as np
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import faiss
import xml.etree.ElementTree as ET
from tqdm import tqdm
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    logger.error("The google-generativeai package is required. Install it with: pip install google-generativeai")
    raise

def fetch_and_parse_sitemap(sitemap_url, base_site_url="https://www.sensorysouk.com"):
    """
    Fetches a sitemap (or sitemap index) and extracts all URLs.
    If it's a sitemap index, it recursively fetches linked sitemaps.

    Args:
        sitemap_url (str): The URL of the sitemap to parse.
        base_site_url (str): The base URL of the site, for constructing absolute URLs if needed.

    Returns:
        set: A set of unique URLs found in the sitemap(s).
    """
    logger.info(f"Fetching sitemap: {sitemap_url}")
    all_found_urls = set()
    try:
        response = requests.get(sitemap_url, timeout=20)
        response.raise_for_status()
        
        # XML content can be sensitive to encoding, try to handle it
        content_type = response.headers.get('content-type', '').lower()
        encoding = response.encoding
        if 'charset' in content_type:
            encoding_from_header = content_type.split('charset=')[-1].split(';')[0].strip()
            if encoding_from_header:
                encoding = encoding_from_header
        
        try:
            xml_content = response.content.decode(encoding or 'utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UnicodeDecodeError with encoding {encoding} for {sitemap_url}, trying 'latin-1'")
            xml_content = response.content.decode('latin-1', errors='replace')

        root = ET.fromstring(xml_content)
        
        # XML Namespaces can make parsing tricky. Common sitemap namespaces:
        namespaces = {
            's': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            # Add other namespaces if you find them in Sensory Souk's sitemap
            'image': 'http://www.google.com/schemas/sitemap-image/1.1'
        }

        # Check if it's a sitemap index (contains <sitemap> tags)
        # or a URL set (contains <url> tags)
        if root.tag.endswith('sitemapindex'):
            logger.debug(f"{sitemap_url} is a sitemap index.")
            for sitemap_entry in root.findall('s:sitemap', namespaces):
                loc_element = sitemap_entry.find('s:loc', namespaces)
                if loc_element is not None and loc_element.text:
                    # Recursively parse linked sitemaps
                    all_found_urls.update(fetch_and_parse_sitemap(loc_element.text.strip(), base_site_url))
        elif root.tag.endswith('urlset'): # More robust
            logger.debug(f"{sitemap_url} is a URL set.")
            for url_entry in root.findall('s:url', namespaces):
                loc_element = url_entry.find('s:loc', namespaces)
                if loc_element is not None and loc_element.text:
                    found_url = loc_element.text.strip()
                    # Ensure URL is absolute (though sitemaps usually have absolute URLs)
                    if not urlparse(found_url).scheme:
                        found_url = urljoin(base_site_url, found_url)
                    all_found_urls.add(found_url)
        else:
            logger.warning(f"Unknown root tag '{root.tag}' in sitemap: {sitemap_url}. Skipping.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        logger.error(f"Error parsing XML from sitemap {sitemap_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing sitemap {sitemap_url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    return all_found_urls

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
                    content = self._extract_text_content(soup, url)
                    
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
        
    def _extract_text_content(self, soup, current_url=""):
        if soup is None:
            logger.error(f"Soup object is None in _extract_text_content for URL: {current_url}")
            return ""

        # Global cleanup
        for element_type in ['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']:
            for element in soup.find_all(element_type):
                if element: element.decompose()

        # Get the main H1 of the page, if it exists, and clean it
        page_h1_element = soup.find('h1', class_=['product__title', 'collection__title', 'page-title', 'main-page-title'])
        if not page_h1_element: 
            page_h1_element = soup.find('h1')
        
        main_h1_text = ""
        if page_h1_element:
            main_h1_text = " ".join(page_h1_element.get_text(strip=True).split())
            if main_h1_text:
                logger.debug(f"Extracted main H1: '{main_h1_text}' from {current_url}")
            else:
                logger.debug(f"Found H1 for {current_url}, but it was empty after stripping.")
        else:
            logger.debug(f"No H1 title found for {current_url}")

        # Determine page type (logic remains similar to your existing one)
        is_product_page_by_content = False
        product_page_target_soup = None
        
        product_desc_heading_element = soup.find('h2', string=lambda text: text and "product description" in text.lower())
        if product_desc_heading_element:
            parent_div_of_heading = product_desc_heading_element.find_parent('div', class_='ungroup-description-tab__heading')
            if parent_div_of_heading:
                actual_content_div = parent_div_of_heading.find_next_sibling('div', class_='mt0 mt--first-child-0 body2')
                if actual_content_div:
                    logger.debug(f"Primary product selector matched for {current_url}")
                    product_page_target_soup = actual_content_div
                    is_product_page_by_content = True
        
        if not is_product_page_by_content:
            potential_product_selectors = [
                {'name': 'div', 'attrs': {'class': 'product-single__description'}},
                {'name': 'div', 'attrs': {'class': 'product__description'}},
                {'name': 'div', 'attrs': {'class': 'rte'}}, 
                {'name': 'article', 'attrs': {'class': 'product-page-content'}},
            ]
            for selector_info in potential_product_selectors:
                found_area = soup.find(selector_info['name'], **selector_info['attrs'])
                if found_area and len(found_area.get_text(strip=True)) > 100: 
                    logger.debug(f"Fallback product selector {selector_info} matched for {current_url}")
                    product_page_target_soup = found_area
                    is_product_page_by_content = True
                    break
        
        path_for_check = ""
        if current_url:
            try: path_for_check = urlparse(current_url).path.lower()
            except Exception as e_parse: logger.error(f"Could not parse current_url '{current_url}': {e_parse}")
        
        is_category_page_by_url = bool(path_for_check and \
                                (path_for_check.startswith('/collections/') or \
                                path_for_check.startswith('/categories/') or \
                                path_for_check.startswith('/category/')))
        
        logger.debug(f"Page type assessment for {current_url}: is_product_page_by_content={is_product_page_by_content}, is_category_page_by_url={is_category_page_by_url}")

        # Text Extraction based on page type
        content_accumulator = []

        if is_product_page_by_content and product_page_target_soup:
            logger.debug(f"Extracting detailed content for PRODUCT page: {current_url}")
            
            # Add the main H1 as the primary title for the product text
            if main_h1_text:
                content_accumulator.append(f"# {main_h1_text}")

            for heading in product_page_target_soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
                prefix = {'h2': '## ', 'h3': '### ', 'h4': '#### ', 'h5': '##### ', 'h6': '###### '}.get(heading.name, '')
                text = " ".join(heading.get_text(strip=True).split())
                # Avoid repeating the main H1 if it's also found as a sub-heading
                if text and (not main_h1_text or main_h1_text.lower() not in text.lower()):
                    content_accumulator.append(f"{prefix}{text}")

            for p_or_li_or_span in product_page_target_soup.find_all(['p', 'li', 'span']):
                text = " ".join(p_or_li_or_span.get_text(strip=True).split())
                noise_phrases = [
                    "add to cart", "decrease quantity", "increase quantity", "open media", "view full details", 
                    "vendor:", "sold out", "buy it now", "ask a question", "share", "regular price", 
                    "sale price", "tax included", "shipping calculated at checkout", "unit price", "/ per", 
                    "you may also like", "recently viewed", "customer reviews", "write a review", 
                    "availability:", "in stock", "quantity", "default title", "select options", "sort by", 
                    "filter by", "view as:", "items per page", "quick view", "be the first one to review", 
                    "money", "qar" 
                ]
                is_just_price = bool(re.fullmatch(r'[\d\.,]+\s*qar', text, flags=re.IGNORECASE))
                if text and len(text) > 10 and not is_just_price and not any(phrase in text.lower() for phrase in noise_phrases):
                    text = re.sub(r'^\s*[\d\.,]+\s*qar\s*-\s*', '', text, flags=re.IGNORECASE).strip()
                    text = re.sub(r'\s*qar\s*$', '', text, flags=re.IGNORECASE).strip()
                    if len(text.split()) > 2: 
                        content_accumulator.append(text)
            
            product_page_extracted_text = "\n\n".join(content_accumulator)

            if main_h1_text:
                final_page_text = f"Product: {main_h1_text}.\n\n{product_page_extracted_text}"
                logger.debug(f"Text for product page {current_url} was prepended with its title.")
            else:
                final_page_text = product_page_extracted_text
                logger.warning(f"Product page {current_url} had no main_h1_text for explicit prepending, using extracted content as is.")

        elif is_category_page_by_url and not is_product_page_by_content: 
            logger.debug(f"Attempting to extract product listings for CATEGORY page: {current_url}")
            if main_h1_text:
                content_accumulator.append(f"# {main_h1_text}")
            
            category_page_main_content_area = soup.find('section', id=lambda x: x and x.endswith('__product-grid'))
            if not category_page_main_content_area:
                category_page_main_content_area = soup.find('main', id='MainContent') or \
                                                soup.find('div', class_='template-collection') or \
                                                soup.find('div', class_='collection_template_section') or \
                                                soup.find('section', class_='shopify-section--collection-template') or \
                                                soup.find('main') or \
                                                soup
            
            if category_page_main_content_area:
                content_accumulator.append("\nThis category page includes the following products:")
                product_card_container_selectors = ["div.product-card__container", "li.grid__item", "div.card-wrapper"]
                product_cards = []
                for sel in product_card_container_selectors:
                    if category_page_main_content_area: product_cards = category_page_main_content_area.select(sel)
                    if product_cards: break
                
                seen_product_titles = set()
                for card_soup in product_cards:
                    title_element = card_soup.select_one("div.mt5 > a.product-card__heading") or \
                                    card_soup.select_one("a.product-card__heading") or \
                                    card_soup.select_one("h3.card__heading > a.full-unstyled-link")
                    if not title_element:
                        h3_heading = card_soup.select_one("h3.card__heading")
                        if h3_heading: title_element = h3_heading.find('a') or h3_heading
                    if not title_element:
                        general_title_el = card_soup.select_one(".product-card__title")
                        if general_title_el: title_element = general_title_el.find('a') or general_title_el
                    
                    if title_element:
                        title_text = " ".join(title_element.get_text(strip=True).split())
                        title_text = re.sub(r'^(Quick view|View)\s*', '', title_text, flags=re.IGNORECASE).strip()
                        title_text = re.sub(r'\s*(\d+(\.\d+)?\s*QAR.*|-\s*\d+(\.\d+)?\s*QAR.*)$', '', title_text).strip() 
                        if title_text and len(title_text) > 2 and title_text not in seen_product_titles:
                            content_accumulator.append(f"* {title_text}")
                            seen_product_titles.add(title_text)
                if not seen_product_titles:
                    content_accumulator.append("(No distinct product titles were automatically extracted from this page's main content using defined selectors.)")
            final_page_text = "\n\n".join(content_accumulator)

        else:
            logger.debug(f"Treating as OTHER page type (general content extraction): {current_url}")
            if main_h1_text:
                content_accumulator.append(f"# {main_h1_text}")
            
            other_main_content = soup.find('main') or soup.find(id='content') or soup.find(id='MainContent') or \
                                soup.find(class_=['content', 'main-content', 'shopify-section', 'page-content', 'rte'])
            if other_main_content:
                # Simplified noise removal for other pages for now
                for p_or_li in other_main_content.find_all(['p', 'li']):
                    text = " ".join(p_or_li.get_text(strip=True).split())
                    if text and len(text.split()) > 4 : content_accumulator.append(text)
            final_page_text = "\n\n".join(content_accumulator)

        # Final cleaning applied to the constructed final_page_text
        full_text = re.sub(r'(\n\s*){3,}', '\n\n', final_page_text).strip()

        if not full_text and main_h1_text and not is_product_page_by_content :
            full_text = f"# {main_h1_text}"
            logger.debug(f"Content for {current_url} reduced to only its H1: '{main_h1_text}'")
        elif not full_text:
            logger.warning(f"Extraction resulted in empty text for URL: {current_url} (after all filtering).")
            
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
                    
                    break 
                    
                except Exception as e:
                    if retry < max_retries:
                        # Implement exponential backoff
                        wait_time = 2 ** retry 
                        logger.warning(f"Embedding API error: {e}. Retrying in {wait_time}s... ({retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to generate embeddings after {max_retries} retries: {e}")
                        # Generate zero vectors as fallback
                        logger.warning(f"Using zero vectors for chunks {i} to {i+len(batch)-1}")
                        # Get embedding dimensionality from model spec or use a default
                        dim = 768 
                        batch_embeddings = [np.zeros(dim).tolist() for _ in batch]
            
            all_embeddings.extend(batch_embeddings)
    
        return all_embeddings

class VectorStore:
    """
    Manages the vector database operations using JSON Lines for metadata.
    """
    def __init__(self, index_path=None, metadata_path=None, rebuild=False): 
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        self.index_path = index_path or os.path.join(data_dir, 'faiss_index.bin')
        self.metadata_path = metadata_path or os.path.join(data_dir, 'metadata.jsonl')
        self.url_mapping_path = os.path.join(data_dir, 'url_mapping.json') 
        
        if rebuild:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                logger.info(f"Rebuild: Deleted existing index file: {self.index_path}")
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
                logger.info(f"Rebuild: Deleted existing metadata file: {self.metadata_path}")
            if os.path.exists(self.url_mapping_path): 
                os.remove(self.url_mapping_path)
                logger.info(f"Rebuild: Deleted existing URL mapping file: {self.url_mapping_path}")

        self.index = None
        self.chunks_metadata = []
        self.url_mapping = {} 
        
        if not rebuild: 
            self.load_index_and_metadata()
        else:
            logger.info("Skipped loading existing index and metadata due to --rebuild flag.")
        
    def load_index_and_metadata(self):
        """Load existing FAISS index, metadata (from JSONL), and URL mapping if available."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            
            if os.path.exists(self.metadata_path):
                self.chunks_metadata = [] 
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): 
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
            if len(metadata_list) != len(embeddings): 
                logger.error(f"Mismatch in input lengths: metadata={len(metadata_list)}, embeddings={len(embeddings)}")
                return False
                
            embeddings_np = np.array(embeddings).astype('float32')
            
            new_metadata_to_add = []
            new_embeddings_to_add_np = []

            if self.index is None:
                # Create new index
                if not embeddings: 
                    logger.warning("No embeddings provided to create a new index.")
                    return False
                embedding_dim = len(embeddings[0])
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.chunks_metadata = metadata_list 
                new_embeddings_to_add_np = embeddings_np
                # Update URL mapping for all new items
                self.url_mapping.update({meta['id']: meta['url'] for meta in metadata_list})
            else:
                # Update existing index: Identify truly new metadata and embeddings
                existing_ids = {meta['id'] for meta in self.chunks_metadata}
                temp_new_embeddings_list = []

                for i, meta in enumerate(metadata_list):
                    if meta['id'] not in existing_ids:
                        self.chunks_metadata.append(meta) 
                        new_metadata_to_add.append(meta) 
                        temp_new_embeddings_list.append(embeddings_np[i])
                        self.url_mapping[meta['id']] = meta['url'] 
                
                if temp_new_embeddings_list:
                    new_embeddings_to_add_np = np.array(temp_new_embeddings_list).astype('float32')
                else:
                    new_embeddings_to_add_np = np.array([]) 

            # Add new embeddings to FAISS index if any
            if new_embeddings_to_add_np.size > 0:
                if new_embeddings_to_add_np.ndim == 1: 
                    new_embeddings_to_add_np = new_embeddings_to_add_np.reshape(1, -1)
                self.index.add(new_embeddings_to_add_np)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save all current metadata as JSON Lines (overwrite existing file)
            with open(self.metadata_path, 'w', encoding='utf-8') as f: 
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
            logger.error(traceback.format_exc()) 
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
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Force delete existing index, metadata, and URL mapping and rebuild from scratch.'
        )

    def handle(self, *args, **options):
        if not settings.GEMINI_API_KEY:
            self.stderr.write(self.style.ERROR("No Google API key found. Set GEMINI_API_KEY in settings."))
            return
            
        crawler = WebCrawler(max_pages=options['max_pages'])
        text_processor = TextProcessor(api_key=settings.GEMINI_API_KEY)
        vector_store = VectorStore(rebuild=options['rebuild'])
        
        urls_to_crawl = []

        # MODIFIED LOGIC TO PRIORITIZE CLI ARGUMENTS
        if options['start_url']:
            urls_to_crawl.append(options['start_url'])
            logger.info(f"Using URL from --url option: {options['start_url']}")
        elif options['urls_file']:
            try:
                with open(options['urls_file'], 'r') as f:
                    urls_to_crawl.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
                logger.info(f"Loaded {len(urls_to_crawl)} URLs from file: {options['urls_file']}")
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error reading URLs file: {e}"))
                return
        else:
            # Fallback to sitemap only if no specific URL or file is provided
            sitemap_url = "https://www.sensorysouk.com/sitemap.xml" 
            logger.info(f"No specific URL or file provided. Attempting to fetch URLs from sitemap: {sitemap_url}")
            sitemap_urls = fetch_and_parse_sitemap(sitemap_url) 

            if sitemap_urls:
                logger.info(f"Found {len(sitemap_urls)} URLs in sitemap(s).")
                for url_from_sitemap in sitemap_urls:
                    path = urlparse(url_from_sitemap).path.lower()
                    # Filter for relevant paths from sitemap
                    if path.startswith('/products/') or path.startswith('/collections/'):
                        urls_to_crawl.append(url_from_sitemap) 
                logger.info(f"Filtered to {len(urls_to_crawl)} relevant URLs from sitemap.")
            else:
                logger.warning("No URLs found from sitemap either.")

        if not urls_to_crawl:
            self.stderr.write(self.style.ERROR("No URLs to process. Please provide a --url, --urls-file, or ensure the sitemap is accessible and contains relevant links."))
            return
            
        urls_to_crawl = sorted(list(set(urls_to_crawl))) 
        
        logger.info(f"Final list of unique URLs to process: {len(urls_to_crawl)}")
        if len(urls_to_crawl) > 20:
             logger.debug(f"Sample URLs to process: {urls_to_crawl[:20]}")
        else:
             logger.debug(f"URLs to process: {urls_to_crawl}")

        self.stdout.write(self.style.SUCCESS(f"Starting processing for {len(urls_to_crawl)} URLs (overall crawl limit per URL start: {options['max_pages']})"))
        
        # Process each URL
        all_chunks = []
        all_metadata = []
        chunk_id_counter = 0
        
        # If we have existing metadata, find the highest ID to avoid collisions
        if not options['rebuild'] and vector_store.chunks_metadata: 
            try:
                # Ensure IDs are treated as integers for max()
                valid_ids = [int(meta['id']) for meta in vector_store.chunks_metadata if 'id' in meta and meta['id'].isdigit()]
                if valid_ids:
                    highest_id = max(valid_ids)
                    chunk_id_counter = highest_id + 1
                else:
                    # If no valid numeric IDs, start based on length or 0 if that's also problematic
                    chunk_id_counter = len(vector_store.chunks_metadata) 
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Could not determine highest_id reliably: {e}. Resetting chunk_id_counter based on metadata length or to 0.")
                chunk_id_counter = len(vector_store.chunks_metadata) 
        elif options['rebuild']:
            logger.info("Rebuilding: chunk_id_counter initialized to 0.")
                
        self.stdout.write(f"Starting chunk ID counter at {chunk_id_counter}")
        
        for url in urls_to_crawl:
            self.stdout.write(f"Processing URL: {url}")
            
            crawler.base_url = url 
            pages_content = crawler.crawl(start_url=url, force=options['force']) 
            
            if not pages_content:
                self.stdout.write(self.style.WARNING(f"No content found or processed for {url} by crawler."))
                continue
                
            self.stdout.write(f"Found {len(pages_content)} page(s) related to {url} to process for vector DB.")
            
            for page_url, page_data in pages_content.items():
                
                # page_text now ALREADY CONTAINS the "Product: Title" prefix 
                # if it was a product page, because _extract_text_content handled it.
                page_text_for_chunks = page_data['text'] 
                
                # Get the original page title (cleaned) for metadata
                # This title from page_data['title'] is what the crawler initially got from <title> or H1
                # and is NOT prepended with "Product: "
                original_page_title_cleaned = " ".join(page_data['title'].split()) 
                
                if not page_text_for_chunks or not page_text_for_chunks.strip():
                    self.stdout.write(self.style.WARNING(f"Skipping empty page content for {page_url} before chunking."))
                    continue
                    
                # No more prepending logic needed here for page_text_for_chunks
                    
                page_chunks = text_processor.chunk_text(
                    page_text_for_chunks, 
                    chunk_size=options['chunk_size'], 
                    chunk_overlap=options['chunk_overlap']
                )
                
                if not page_chunks:
                    self.stdout.write(self.style.WARNING(f"No chunks created for {page_url}"))
                    continue
                    
                for chunk_index, chunk_text_content in enumerate(page_chunks): 
                    chunk_metadata = {
                        'id': str(chunk_id_counter),
                        'url': page_url,
                        'title': original_page_title_cleaned, 
                        'text': chunk_text_content,       
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': chunk_index 
                    }
                    
                    all_chunks.append(chunk_text_content) 
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