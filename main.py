from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "answer": f"You asked: {request.question}"
    }
