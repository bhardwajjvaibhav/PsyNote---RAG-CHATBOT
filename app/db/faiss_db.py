import faiss
import numpy as np
import os 
import pickle
from typing import List, Dict


class FAISSVectorStore:

    def __init__(self,dim=384,index_path:str="faiss.index",metadata_path:str="faiss_metadata.pkl"):
        self.dim=dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index =faiss.IndexFlatL2(dim)
        self.metadata=[]

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()

    
    def add_embeddings(self, items:List[Dict]):

        vectors=np.array([item["embedding"] for item in items]).astype("float32")
        self.index.add(vectors)
        self.metadata.extend([
            {"text":item["text"],"metadata":item["metadata"]} for item in items
        ])

        self.save()


    def search(self,query_vector:np.ndarray, top_k:int=5)->List[Dict]:
        query_vector=query_vector.astype("float32").reshape(1,-1)
        distances,indices=self.index.search(query_vector,top_k)

        results=[]

        for idx in indices[0]:
            if idx<len(self.metadata):
                results.append(self.metadata[idx])
        return results
      

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path,'wb') as f:
            pickle.dump(self.metadata,f)


    def load(self):
        self.index=faiss.read_index(self.index_path)
        with open(self.metadata_path,'rb') as f:
            self.metadata=pickle.load(f)

