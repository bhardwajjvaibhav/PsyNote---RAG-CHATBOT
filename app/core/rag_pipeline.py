import os
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from app.services.embedder import Embedder
from app.db.faiss_db import FAISSVectorStore

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL")
        self.model = os.getenv("OPENROUTER_MODEL")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.embedder = Embedder()
        self.vector_store = FAISSVectorStore()

    def get_answer(self, question: str, chat_history: List[Dict[str, str]] = None, top_k: int = 5) -> str:
        # Step 1: Embed the question
        query_vec = self.embedder.model.encode([question])[0]

        # Step 2: Search relevant context chunks from vector DB
        results = self.vector_store.search(np.array(query_vec), top_k=top_k)
        context_chunks = [r["text"] for r in results[:2]]
        context = "\n\n".join(context_chunks)

        # Step 3: Format chat history (if provided)
        chat_history_str = ""
        if chat_history:
            recent_history=chat_history[-2:]
            for turn in recent_history:
                chat_history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        # Step 4: Build the full prompt
        prompt = f"""You are a helpful and empathetic mental health assistant. Use the following context and chat history to answer the user's latest question thoughtfully.

Chat History:
{chat_history_str}

Retrieved Context:
{context}

User: {question}
Assistant:"""

        # Step 5: Call the OpenRouter API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content
