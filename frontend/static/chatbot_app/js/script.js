/**
 * Chat interface JavaScript functionality
 * 
 * Handles user input submission, displays messages in the chat interface,
 * communicates with the backend API, and manages the UI state.
 */

// Wait for DOM to be fully loaded before attaching event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Get required DOM elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatbox = document.getElementById('chatbox');
    const sendButton = document.getElementById('send-button');
    
    // Get CSRF token from the cookie for secure POST requests
    function getCSRFToken() {
        // Django sets a csrf token in a cookie named 'csrftoken'
        const name = 'csrftoken';
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
        return '';
    }
    
    // Add a message to the chat display
    function addMessageToChat(message, isUser) {
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        
        // Add message content
        const messageContent = document.createElement('p');
        messageContent.textContent = message;
        messageDiv.appendChild(messageContent);
        
        // Add to chatbox and scroll to bottom
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    // Show typing indicator while waiting for response
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        const dots = document.createElement('p');
        dots.textContent = 'Thinking...';
        typingDiv.appendChild(dots);
        
        chatbox.appendChild(typingDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Handle form submission
    chatForm.addEventListener('submit', function(event) {
        // Prevent the default form submission behavior
        event.preventDefault();
        
        // Get user message and trim whitespace
        const message = userInput.value.trim();
        
        // Don't do anything if the message is empty
        if (!message) {
            return;
        }
        
        // Display user message in the chat
        addMessageToChat(message, true);
        
        // Clear input field
        userInput.value = '';
        
        // Disable input while waiting for response
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Show typing indicator
        showTypingIndicator();
        
        // Get CSRF token for secure POST request
        const csrfToken = getCSRFToken();
        
        // Send request to backend
        fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            // Check if response is ok (status in 200-299 range)
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Display bot response
            if (data.error) {
                addMessageToChat('Sorry, there was an error processing your request: ' + data.error, false);
            } else {
                addMessageToChat(data.response, false);
            }
        })
        .catch(error => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Display error message
            addMessageToChat('Sorry, there was a problem connecting to the server. Please try again.', false);
            console.error('Error:', error);
        })
        .finally(() => {
            // Re-enable input regardless of outcome
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        });
    });
    
    // Focus input field when page loads
    userInput.focus();
});