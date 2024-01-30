import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from transformers import pipeline

# Log in to Hugging Face and grant authorization to Hugging Chat
sign = Login('shravaniphadol48@gmail.com', 'Shravani2003')
cookies = sign.login()

# Save cookies to the local directory
cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)

# Create a ChatBot
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

# Initialize sentiment analysis pipeline with a specific model
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Streamlit app
st.title("UNISPECTRA BOT")
st.sidebar.title("Chat History")

# Initialize chat history
chat_history = []

# Function to update and display chat history
def update_chat_history(user_input, bot_response, sentiment_label):
    chat_history.append({
        'user': {'text': user_input, 'sentiment': sentiment_label},
        'bot': {'text': bot_response, 'sentiment': sentiment_label}
    })
    st.sidebar.text(f"User ({sentiment_label}): {user_input}")
    st.sidebar.text(f"Bot ({sentiment_label}): {bot_response}")

# Streamlit input and output components
user_input = st.text_input(
    "Chat with UNISPECTRA Bot for emotional and health support:")

if st.button("Submit"):
    if user_input:
        # Get sentiment analysis result
        sentiment_result = sentiment_analysis_pipeline(user_input)
        sentiment_label = sentiment_result[0]['label']

        # Get response from the ChatBot
        bot_response = chatbot.query(user_input)

        # Update and display chat history
        update_chat_history(user_input, bot_response, sentiment_label)

        # Display bot response
        st.text(f"ChatWiz ({sentiment_label}): {bot_response}")

# "Delete Chat" button
if st.button("Delete Chat"):
    chat_history = []
    st.sidebar.text("Chat history deleted.")

# Display chat history in the sidebar
st.sidebar.title("Chat History")
for entry in chat_history:
    st.sidebar.text(
        f"User ({entry['user']['sentiment']}): {entry['user']['text']}")
    st.sidebar.text(
        f"Bot ({entry['bot']['sentiment']}): {entry['bot']['text']}")
