# SanConnect Product Assistant

SanConnect Product Assistant is an internal chat-based tool for querying Sensory Souk product information. It serves as an MVP (minimum viable product) to enable Sanco Group employees to ask natural-language questions about Sensory Souk products and receive accurate, fact-based answers. The system is grounded entirely in Sensory Souk's product data (scraped from the website), ensuring responses are drawn from real product details rather than hallucinated content. In short, this assistant aims to be the company's go-to source for Sensory Souk product info, handling follow-up questions via a simple web chat interface.

## Core Features

1. The assistant's knowledge is built by crawling and parsing the Sensory Souk website. It indexes all product pages so that questions like "Describe [Product X]" can be answered with information pulled directly from the site.

2. User queries are processed through a Retrieval-Augmented Generation pipeline. Google's Gemini LLM interprets the question, while the system retrieves relevant text chunks from the product database to ground the answer.

3. All responses are strictly based on the scraped product data. This design choice prevents the AI from "making up" information – if the answer can't be found in the product content, the assistant will acknowledge its limits.

4. The assistant maintains a short conversation history (sent via the frontend) to handle follow-up questions sensibly. This basic chat memory lets it relate answers to the previous user turn.

5. A minimal web frontend allows users to chat with the assistant in any browser. Markdown formatting (via marked.js) is used so answers can include bullets or bold text for clarity.

6. Each chat exchange is logged to a SQLite database (the ChatLog model). This logging is mainly for analysis and future improvement of the assistant.

## Architecture & Tech Stack

- The backend is a Django app that handles both the chat API and the knowledge-base updates.

- A lightweight SQLite database stores chat logs and is used during development/testing.

- Google's Gemini models are used for both chat and embeddings. Specifically, a gemini-2.0-flash (or similar) chat model generates answers, and an embedding-001 model encodes text chunks.

- The requests library and BeautifulSoup are used to crawl and parse Sensory Souk's product pages. The sitemap is parsed via xml.etree.ElementTree for full-site crawling.

- Text from the website is cleaned, chunked, and converted into embeddings. These are stored in a FAISS vector index (faiss_index.bin) for fast similarity search.

- After retrieving candidate text chunks from FAISS, the system uses scikit-learn's TF-IDF vectorizer to measure similarity and re-rank results.

- A simple HTML/CSS/JavaScript page serves as the chat UI. Responses (in Markdown) are rendered with the [marked.js] library.

- Standard Python virtual environments (venv) are used to manage packages, and the code is maintained in this GitHub repository.

## How It Works (RAG Flow)

### Knowledge Base Construction

A Django management command (`manage.py update_vectordb`) crawls the Sensory Souk site and builds the retrieval index. Pages can be seeded via specific URLs or by using the site's sitemap. Each product page is fetched (using requests), parsed for key text (via BeautifulSoup and custom extractors), and split into smaller chunks. Each chunk is embedded (using Gemini's embedding model) and added to the FAISS index, with its source metadata recorded in a JSONL file.

### Query Processing

When a user submits a question in the web UI, the backend's process_message function executes several steps:

1. It extracts any specific product name mentioned in the query.
2. It analyzes the query intent (e.g. product detail, comparison, category info).
3. It generates multiple "targeted" search queries based on the intent.
4. For each targeted query, it obtains an embedding and searches the FAISS index for matching text chunks (applying URL filters if a product is identified).
5. It collects a pool of candidate chunks from these searches.

### Result Selection & Answer Generation

The candidate chunks are re-ranked using a custom process. It scores chunks by TF-IDF similarity to the query, information density, and intent-specific keywords (penalizing very short or irrelevant snippets). The top-N chunks are then formatted with source tags and passed, along with the user's conversation history and question, into a single system prompt for Gemini. This prompt includes a dynamic system instruction (based on intent) and the selected context chunks. Gemini generates a final answer, which is lightly post-processed and then returned to the user. The entire interaction (user message, retrieved chunks, and bot response) is logged in the SQLite ChatLog for review.

## Setup Instructions

1. Clone the Repository:
   ```
   git clone https://github.com/your-org/InternalGuideAI.git
   cd InternalGuideAI/backend
   ```
   (Adjust the path if your manage.py is located elsewhere.)

2. Create a Python Virtual Environment:
   ```
   python -m venv venv
   ```
   
3. Activate it:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`

4. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Configure Environment Variables:
   In the backend directory (same level as manage.py), create a file named `.env`. Add your Google Gemini API key:
   ```
   GEMINI_API_KEY="your_api_key_here"
   ```
   Ensure settings.py is set up to load this key (for example, via python-dotenv). Also verify that Django's SECRET_KEY is configured (Django can auto-generate one for development).

6. Apply Database Migrations:
   ```
   python manage.py migrate
   ```

7. Build/Update the Knowledge Base:
   - To update from specific URLs (e.g. in a file data/seed_urls.txt):
     ```
     python manage.py update_vectordb --rebuild --force --urls-file data/seed_urls.txt
     ```
   - To update using the sitemap (full site crawl):
     ```
     python manage.py update_vectordb --rebuild --force --max-pages 500
     ```
   (The --rebuild flag clears old data; --force re-fetches cached pages. Adjust --max-pages as needed.)

8. Run the Server:
   ```
   python manage.py runserver
   ```
   Then open http://127.0.0.1:8000/ in your browser to start chatting with the assistant.

## Limitations & Future Work

This MVP has a basic UI (no advanced styling) and no user accounts. Conversation state is kept only in the current session (no long-term memory). It cannot yet handle very complex multi-step queries or automatically link to external tools (e.g. inventory systems). The focus is on single-product Q&A; comparing multiple products or answering broad category questions is limited.

Planned improvements include better product name recognition (using fuzzy matching or expanded NLP) and continuously tuning the chunk re-ranking process. We aim to improve handling of ambiguous queries and to extend support for robust comparisons and category-level inquiries. Other ideas include enhanced error handling and feedback, scaling to larger catalogs, and building an evaluation framework for systematic testing.

## Acknowledgments

This project was developed by Harish Kumbar as part of the Sanco Group's AI initiatives. It was inspired by the broader "Sanco Connect" vision for a centralized AI assistant spanning multiple Sanco companies. The author thanks the Sensory Souk product team for providing data access and the Sanco Group technology mentor for guidance.