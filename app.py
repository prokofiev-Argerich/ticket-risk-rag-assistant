from fastapi import FastAPI
from pydantic import BaseModel
from service import analyze_ticket

app = FastAPI(title="Ticket Risk RAG Assistant")


class AnalyzeRequest(BaseModel):
    text: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return analyze_ticket(req.text, top_k=req.top_k)
