from app.services.parser import load_document
from app.services.chunker import chunk_text
from app.services.embedder import Embedder 
from app.db.faiss_db import FAISSVectorStore
from app.core.rag_pipeline import RAGPipeline

import os
from dotenv import load_dotenv

load_dotenv()

# Global history maintained throughout conversation
chat_history = []

def initialize_vector_db(file_path: str):
    print("📄 Reading document...")
    text = load_document(file_path)
    print("📂 Type of loaded text:", type(text))

    print("✂️ Chunking document...")
    chunks = chunk_text(text)
    print(f"🧩 Total chunks: {len(chunks)}")

    print("🔗 Creating embeddings...")
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    print("📦 Saving to FAISS DB...")
    store = FAISSVectorStore()
    store.add_embeddings(embedded_chunks)

    print("✅ Document processed and stored in vector DB.")


def chat_with_bot():
    rag = RAGPipeline()

    while True:
        question = input("You 🧠: ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        answer = rag.get_answer(question, chat_history=chat_history)

        chat_history.append({"user": question, "assistant": answer})
        print(f"Bot 🤖: {answer}\n")


if __name__ == "__main__":
    FILE_PATH = "data/journal_entries.txt"  # replace with actual path
    initialize_vector_db(FILE_PATH)
    chat_with_bot()
