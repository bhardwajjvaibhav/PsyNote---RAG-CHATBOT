from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List,Dict
from app.core.rag_pipeline import RAGPipeline

router=APIRouter()
rag=RAGPipeline()
chat_history:List[Dict[str,str]] = []

class ChatRequest(BaseModel):
    question:str


class ChatResponse(BaseModel):
    answer:str

@router.post("/chat",response_model=ChatResponse)
 
def chat(request:ChatRequest):
    try:
        answer=rag.get_answer(request.question, chat_history=chat_history)
        chat_history.append({"user":request.question, "assistant":answer})
        return ChatResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))