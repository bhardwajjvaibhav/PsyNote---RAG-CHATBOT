from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


class Embedder:

    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    
    def embed_chunks(self, chunks:List[Dict])->List[Dict]:

        texts=[chunk["content"] for chunk in chunks]

        embeddings=self.model.encode(texts,show_progress_bar=False)
        return[
            {
                "embedding":emb.tolist(),
                "text":chunk["content"],
                "metadata":chunk.get("metadata",{})
            }
            for emb, chunk in zip(embeddings, chunks)
        ]