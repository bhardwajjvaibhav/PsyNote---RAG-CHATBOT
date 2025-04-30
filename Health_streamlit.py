import streamlit as st
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model_name = os.getenv("OPENROUTER_MODEL")  # e.g., "deepseek/deepseek-r1-0528-qwen3-8b:free"

# Initialize OpenRouter client
client = OpenAI(api_key=api_key, base_url=base_url)

# Embedder setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# App state
st.set_page_config(page_title="Mental Health Assistant")
st.title("🧠 Mental Health Assistant")
st.write("Upload your journal text file and chat with your mental health assistant.")

# Upload journal
uploaded_file = st.file_uploader("📂 Upload a journal file (.txt)", type="txt")

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# Process uploaded file
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.session_state.document_chunks = text.split("\n\n")  # Split into context chunks

# Chat input
user_input = st.text_input("💬 Ask a question about your journal:")
if user_input and st.session_state.document_chunks:
    # Embed question
    query_vec = embedder.encode([user_input])[0]

    # Embed and score chunks
    doc_vecs = embedder.encode(st.session_state.document_chunks)
    scores = np.dot(doc_vecs, query_vec)

    # Top-2 context
    top_indices = scores.argsort()[-2:][::-1]
    top_context = "\n\n".join([st.session_state.document_chunks[i] for i in top_indices])

    # Format chat history
    history = ""
    for turn in st.session_state.chat_history[-2:]:
        history += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    # Final prompt
    prompt = f"""You are a helpful and empathetic mental health assistant. Use the following journal content and chat history to answer the user's latest question thoughtfully.

Chat History:
{history}

Journal Context:
{top_context}

User: {user_input}
Assistant:"""

    # Call OpenRouter
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )

    reply = response.choices[0].message.content.strip()
    st.session_state.chat_history.append({"user": user_input, "assistant": reply})

# Show conversation
st.markdown("### 🗣️ Conversation")
for turn in st.session_state.chat_history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")
