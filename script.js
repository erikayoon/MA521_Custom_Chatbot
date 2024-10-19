// Grab elements from the DOM
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Function to add messages to the chat box
function addMessage(message, isBot = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message');
    if (isBot) {
        messageDiv.classList.add('bot-message');
    } else {
        messageDiv.classList.add('user-message');
    }
    messageDiv.innerText = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
}

// Handle sending the message
sendBtn.addEventListener('click', async () => {
    const question = userInput.value.trim();

    if (!question) return; // Don't send if the input is empty

    // Add the user's message to the chat
    addMessage(question, false);
    userInput.value = ''; // Clear the input box

    // Send the question to the backend API and get a response
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        const botResponse = data.answer || 'Sorry, I could not process your question.';

        // Add the bot's response to the chat
        addMessage(botResponse, true);
    } catch (error) {
        addMessage('Error: Unable to contact server.', true);
    }
});

// Handle "Enter" key press to send message
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});
