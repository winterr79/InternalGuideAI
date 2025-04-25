# InternalGuideAI - Sensory Souk Product Assistant MVP

## 1. What's the Goal Here?

Getting quick, accurate information about specific products can sometimes be a challenge across Sanco Group's diverse verticals. This project is a first step – a Minimum Viable Product (MVP) – towards building a helpful internal AI assistant.

The main goal of this MVP is to test the core concept and technical feasibility of using a conversational AI (powered by Google Gemini via its API) within a simple Django web application to provide specific information, starting with a few products from the Sensory Souk catalog.

I aim to show:
* How an AI can act as a knowledgeable internal guide.
* That it can understand varied employee questions (natural language).
* That I can make it provide answers based only on specific company data (product info), ensuring accuracy.
* How it can handle basic conversational follow-ups.

## 2. What This MVP Does (and Doesn't Do)

To keep this initial version focused and achievable, I've defined a clear scope:

### Core Functionality:

This MVP acts as a Sensory Souk Product Information Assistant. You can ask it questions about 3-5 specific products (like the Wooden Busy Board, Time Tracker, Nee Doh balls - the exact list is defined in the data).

* It uses the Gemini API to figure out what product you're asking about, even if you phrase your question differently.
* Critically, it generates answers only from the product information I provide it (in a simple JSON file). This prevents the AI from making things up (hallucinating) and keeps  the information accurate.
* It remembers the last product discussed in the current chat session, so you can ask simple follow-up questions like "What age is it for?".
* If you ask about something outside the known products, it will politely let you know it doesn't have that information yet.
* Interaction happens through a basic web page.
* Conversations are logged to a simple database (SQLite) for debugging and future analysis.

### Key Limitations (What's NOT Included Yet):

To keep the focus tight for this initial version, several features are not included:

* A polished, complex user interface (UI).
* Deep, long-term memory (it only remembers context within the current session).
* Personalization based on user roles or complex context understanding.
* Sophisticated error handling beyond basic API/server issues.
* Integrations with other tools like Microsoft Teams or mind-mapping software.
* Complex tasks like step-by-step troubleshooting (the focus is purely on product info retrieval).
* Connections to live inventory, order systems, or other complex databases.

## 3. The Tech Stack

I'm using a fairly standard setup for this kind of web application:

* Python with the Django Framework
* SQLite (simple, file-based database included with Django, used mainly for logging in this MVP)
* Google Gemini API (accessed via standard REST API calls)
* Python requests library
* Basic HTML, CSS, and JavaScript for the chat interface
* Python Virtual Environment (venv) to manage dependencies
* Git and GitHub

## 4. How It Works (The Basic Flow)

1. A user types a message into the simple web interface.
2. The frontend JavaScript sends this message to our Django backend.
3. The Django view function receives the message. It looks at the user's current session to retrieve the recent chat history (to understand context).
4. It tries to figure out if the user's message is about one of the known Sensory Souk products (using simple keyword checks or potentially asking the AI).
5. If it's about a known product:
   - The view loads the relevant product details from our static data file (.json).
   - It constructs a detailed prompt for the Gemini API. This prompt includes: instructions for the AI (act as assistant, use only provided data), the product data itself (grounding context), the recent chat history, and the user's actual question.
   - It calls the Gemini API using the requests library.
6. If it's not about a known product:
   - The view prepares a standard "Sorry, I only know about these products..." response.
7. The view gets the response back from the Gemini API (or uses the standard out-of-scope response).
8. It updates the chat history stored in the user's session.
9. It logs the user's message and the bot's response to the SQLite database.
10. It sends the bot's response back to the frontend JavaScript.
11. The JavaScript displays the bot's message in the chat window.

## 5. Example Conversation

Here's how an interaction might look:

**Bot:** Hi! I'm the Internal Guide AI. I currently have information about the Wooden Busy Board, Time Tracker Visual Timer, and Nee Doh Stress Balls. How can I help?

**User:** Tell me about the busy board

**Bot:** The Wooden Busy Board is a 30x23 cm timber board designed for children 3 years and older. It helps enhance fine motor control, problem-solving, and patience using various latches and hatches. It features non-toxic paint but requires adult supervision due to small parts.

**User:** what age is it recommended for?

**Bot:** It's recommended for ages 3 years and up, but please note it has small parts and requires adult supervision.

**User:** Thanks

**Bot:** You're welcome! Can I help with information on the other products?

**User:** What's the capital of Qatar?

**Bot:** Sorry, I only have information about the Wooden Busy Board, Time Tracker Visual Timer, and Nee Doh Stress Balls right now.

## 6. Getting Started (Setup & Running)

*(Detailed setup instructions will be added here, covering: cloning the repository, setting up the Python virtual environment, installing dependencies from requirements text, creating a file to store environment varibales and adding your GEMINI_API_KEY, running Django database migrations, and starting the development server using python runserver.)*