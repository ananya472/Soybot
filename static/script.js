async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();
    if (!message) return;

    const chatBox = document.getElementById("chat-box");

    // Add user message
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "user-message");
    userMessage.innerText = "You: " + message;
    chatBox.appendChild(userMessage);

    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // Fetch response
    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: message })
    });

    const data = await response.json();

    // Add bot response
    const botMessage = document.createElement("div");
    botMessage.classList.add("message", "bot-message");
    botMessage.innerText = "Bot: " + data.response;
    chatBox.appendChild(botMessage);

    chatBox.scrollTop = chatBox.scrollHeight;
}
