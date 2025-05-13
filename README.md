# SanConnect Product Assistant

## The Big Idea: An Internal Product Helper

Ever find yourself digging for product details across Sanco Group's different businesses? It can be a bit of a maze! This project, "SanConnect Product Assistant," is my initial crack at building a smart internal helper to make that easier. Think of this as the first version (an MVP) to see if we can get an AI to be our go-to for product info.

For this MVP, I'm focusing on:
*   Seeing if a conversational AI (using Google's Gemini) can actually work as a quick internal guide.
*   Making sure it can understand normal, everyday questions from employees.
*   The really important bit: **grounding its answers strictly in our company's product data** (scraped from the Sensory Souk website). This means no made-up stuff – just the facts!
*   Getting it to handle simple back-and-forth conversation, like follow-up questions.

## What's In This Version (And What's On the Wishlist)

To get this off the ground quickly, I've kept the first version (MVP) pretty focused. Here's what it can do right now:

*   **Your Sensory Souk Product Go-To:** It builds its knowledge from the Sensory Souk website, aiming to answer questions about products found there.
*   **Understands Your Questions (RAG Powered):** Thanks to Google's Gemini and a Retrieval-Augmented Generation (RAG) pipeline, you can ask about products in different ways. The system tries to find relevant text from the website data to help form an answer.
*   **Fact-Based Answers Only:** This is super important! The assistant *only* uses the product info scraped from the website. This keeps the answers accurate and stops the AI from guessing or making things up (no "hallucinations" here!).
*   **Basic Chat Memory:** It can use recent conversation history (sent by the frontend) to understand follow-up questions.
*   **Knows Its Limits:** If you ask about something outside its current knowledge base or if the retrieved information is insufficient, it should politely tell you.
*   **Simple Web Interface:** You chat with it through a webpage.
*   **Learning Log:** I'm logging conversations to a basic SQLite database. This is mainly for me to see what questions are being asked, how it's doing, and where to improve it.

**What's Not In This MVP (Future Ideas!):**

To keep things manageable for this first pass, I've left out some of the fancier stuff for now:
*   A super-polished, pixel-perfect UI (it's functional!).
*   User accounts or remembering things across different browser sessions without a persistent conversation ID from the client.
*   Deep understanding of very complex, multi-step queries or user roles.
*   Super-fancy error messages (it handles the basics).
*   Direct hookups to tools like Teams or mind-mapping software (though that's a cool idea for later!).
*   Helping with complex troubleshooting steps for products.
*   Live links to inventory, order systems, or other big databases.
*   Advanced comparison between more than two products simultaneously.
*   Comprehensive category-level Q&A (current focus is on product-specific details).

## The Tech Behind It

Here's what's powering this assistant:

*   **Python & Django:** The backbone of the web application.
*   **SQLite:** A simple, file-based database that comes with Django, used for `ChatLog`.
*   **Google Gemini API:** The brains for understanding questions and generating answers (specifically using models like `gemini-2.0-flash` for chat and `models/embedding-001` for embeddings).
*   **Data Ingestion & RAG (Retrieval-Augmented Generation):**
    *   `requests` and `BeautifulSoup4` for crawling and parsing product information from the Sensory Souk website.
    *   `xml.etree.ElementTree` for parsing sitemaps.
    *   `faiss-cpu` for creating and searching a vector index of the product information.
    *   `scikit-learn` (specifically `TfidfVectorizer`) for text analysis in the re-ranking process.
    *   Custom Python logic for text chunking, intent analysis, multi-query generation, and re-ranking retrieved chunks.
*   **Frontend:** Basic HTML, CSS, and JavaScript.
    *   `marked.js` for rendering Markdown in the bot's responses.
*   **Dev Environment:** Standard Python virtual environment (`venv`) for keeping dependencies tidy, and Git/GitHub for version control.

## How It All Connects (The RAG Flow)

Here's a peek under the hood at how your questions get answered:

1.  **Knowledge Base First (The `update_vectordb` command):**
    *   A Django management command (`manage.py update_vectordb`) is run to build/update the knowledge base.
    *   It can start from specific URLs, a file of URLs, or the website's sitemap.
    *   It crawls pages using `requests`.
    *   `BeautifulSoup` parses HTML, and custom logic in `_extract_text_content` pulls key textual information (product descriptions, features, category product lists).
    *   Text is cleaned and broken into smaller, manageable chunks by `TextProcessor`.
    *   Each chunk is converted into a numerical embedding using `models/embedding-001`.
    *   Embeddings are stored in a FAISS vector index (`faiss_index.bin`).
    *   The corresponding text chunks and metadata (URL, page title, chunk index, etc.) are saved in `metadata.jsonl`.
    *   A `url_mapping.json` and `content_cache.json` are also used to manage crawl state and efficiency.

2.  **Chatting with the Assistant (The Web App via `views.py`):**
    *   You type your question into the web interface.
    *   JavaScript sends your message (and potentially conversation history and ID) to the Django backend (`/api/chat/`).
    *   **Inside `process_message` in `chatbot_app/views.py`:**
        a.  `extract_potential_product_name`: Tries to identify product(s) in your query.
        b.  `analyze_query_intent`: Determines if your query is about a specific product aspect, a general product overview, a comparison, a category, or a general question.
        c.  `generate_targeted_queries`: Creates several specific search queries based on this intent.
        d.  `multi_query_retrieval`: For each targeted query, it generates an embedding and searches the FAISS index. It applies URL filters if a specific product is identified. It gathers a candidate set of text chunks.
        e.  `select_final_chunks` (which uses `re_rank_chunks`): This crucial step re-ranks the candidate chunks. It uses TF-IDF similarity to your query, calculated information density of chunks, intent-specific keywords, and penalties for very short/uninformative chunks. The top N (e.g., 6) chunks are selected.
        f.  `format_chunks_for_context`: The selected chunks are formatted with source information and special tags (e.g., `[Direct Info for Product X]`).
        g.  `generate_llm_response`: A detailed prompt is constructed for the Gemini chat model (`gemini-1.5-flash-latest`). This includes:
            *   A dynamic system instruction (from `get_system_prompt`) based on the query intent.
            *   The formatted retrieved context.
            *   Conversation history.
            *   Your actual question.
        h.  Gemini generates an answer based on this comprehensive prompt.
        i.  `post_process_response`: Minor cleanup of the LLM's text.
        j.  The interaction is logged to the SQLite database (`ChatLog` model).
        k.  The bot's response is sent back to the frontend.
    *   The frontend JavaScript uses `marked.js` to render Markdown in the response (like bullet points or bold text).

## What a Chat Might Look Like

**You:** `Hi SanConnect!`  
**Bot:** `Hello! I'm the SanConnect Product Assistant. I can help you with information about our Sensory Souk products. What would you like to know?`

**You:** `Describe Space Explorers.`  
**Bot:**
The **Space Explorers** body stocking is a sensory tool designed to stretch the imagination. It is an ultimate body stocking made of 4-way stretch spandex that provides proprioceptive input. The snug fabric hugs the body, offering resistance and soothing benefits for a sensory diet, helping kids calm down. It is also lightweight and travels easily.

It helps children develop spatial and body awareness, muscle strength, and motor planning. It also supports sensory-seeking kids, teens, and adults with ADHD, sensory integration disorder, or autism. It can be used when you need a little downtime, immersing yourself in the cozy, stretchy, and supremely comfortable material. It encourages imaginative play, such as pretending to be animals or objects.

The Space Explorers sack is available in the following sizes:
*   **X-Small (Purple MW6233):** 30"L x 22"W, ages 2-4
*   **Small (Red MW5639):** 40"L x 26"W, ages 3-5
*   **Medium (Green MW5640):** 47"L x 27"W, ages 6-8
*   **Large (Orange MW5641):** 56"L x 28"W, ages 9-12
*   **X-Large (Blue MW5642):** 60"L x 28"W, ages 12-adult

It weighs approximately 7 oz and is made of 93% polyester, 5% spandex, and 100% polyester hook and loop closures.
*(Source: Space Explorers – SensorySouk.com - https://www.sensorysouk.com/products/space-explorers)*

**You:** `What conditions is the Spio Classic TLSO ideal for?`  
**Bot:** `The Spio Classic TLSO is ideal for those with Hypotonia, Cerebral Palsy, Down Syndrome, Muscular Atrophy, Athetosis, and Sensory Processing Disorder. *(Source: Spio Classic TLSO – SensorySouk.com - https://www.sensorysouk.com/products/spio-compression-clothing)*`

**You:** `Thanks!`  
**Bot:** `You're welcome! Is there anything else I can help you with regarding Sensory Souk products?`

## Setting Up Your Own SanConnect Assistant

Want to get this running locally? Here's the drill:

1.  **Clone the Repo:**
    ```bash
    git clone [URL_OF_YOUR_GITHUB_REPO]
    cd InternalGuideAI/backend
    ```
    *(This assumes your `manage.py` is in the `InternalGuideAI/backend` subdirectory of your Git clone. Adjust if your structure is different.)*

2.  **Create & Activate a Python Virtual Environment:**
    It's best practice to keep project dependencies isolated.
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    All the Python packages needed are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   In the `InternalGuideAI/backend` directory (alongside `manage.py` and `settings.py`), create a file named `.env`.
    *   Add your Google Gemini API key to this file. Your `settings.py` should be configured to load this.
        ```env
        GEMINI_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
        # Ensure your settings.py loads this, e.g., using python-dotenv
        # Example for settings.py:
        # from dotenv import load_dotenv
        # load_dotenv()
        # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        ```
    *   Ensure your `settings.py` has a `SECRET_KEY`. For development, Django generates one. For production, this should be a strong, unique key kept secret.

5.  **Apply Database Migrations:**
    This sets up the necessary database tables (including for `ChatLog`).
    ```bash
    python manage.py migrate
    ```

6.  **Build/Update the Knowledge Base:**
    Before the chatbot can answer questions accurately, you need to crawl the website and build the vector index.
    *   **To process specific URLs (good for initial testing or small updates):**
        Create a file (e.g., `data/seed_urls.txt`) with one product/category URL per line. Then run:
        ```bash
        python manage.py update_vectordb --rebuild --force --urls-file data/seed_urls.txt
        ```
        *(`--rebuild` deletes old data, `--force` re-crawls even if cached. Use `--max-pages` to limit crawl depth from each seed URL if they are category pages.)*
    *   **To process the whole site via sitemap (for a full build):**
        ```bash
        python manage.py update_vectordb --rebuild --force --max-pages 500 
        ```
        *(This will use the sitemap defined in `update_vectordb.py`. Adjust `--max-pages` as needed. This can take a while!)*

7.  **Run the Django Development Server:**
    ```bash
    python manage.py runserver
    ```

8.  **Chat!**
    Open your web browser and go to `http://127.0.0.1:8000/`. You should see the SanConnect Product Assistant ready for your questions!

## Future Enhancements & Considerations

*   **Improving Product Name Extraction:** This is key for reliability. Exploring fuzzy matching against a known product list or more advanced NLP techniques could help.
*   **Refining Re-ranking Logic:** Continuously tuning the weights and heuristics in `re_rank_chunks` based on test cases.
*   **Handling Ambiguity:** Better strategies for when a user's query is vague or could refer to multiple products/aspects.
*   **Advanced Comparison:** More robust logic for comparing 2+ products across multiple aspects.
*   **Category Q&A:** Fully implementing and testing questions about product categories.
*   **Error Handling & User Feedback:** More graceful error handling and providing users with clearer feedback when information can't be found.
*   **Scalability:** For very large product catalogs, optimizing the retrieval and re-ranking steps will be important.
*   **Evaluation Framework:** Implementing a systematic way to test the bot's accuracy and helpfulness on a predefined set of questions.

---
This project is a learning journey into building effective RAG systems. Contributions and suggestions are welcome!