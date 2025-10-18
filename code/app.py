from fastapi import FastAPI
from .model import ChatRequest






app = FastAPI()

@app.get("/chat")
def chat(request: ChatRequest) -> str:
    return request.prompt

