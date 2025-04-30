from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]  # 👈 FIXED: no empty string
    )
    chunks = splitter.split_text(text)
    return [{"content": chunk, "metadata": {"length": len(chunk)}} for chunk in chunks]
