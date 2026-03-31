import jieba
import torch
from rag import retrieve_docs
from templates import build_reply_suggestion
from predict_core import predict_tokens

MODEL_PATH = "model.pt"


def analyze_ticket(text: str, top_k: int = 3):
    tokens = [w for w in jieba.lcut(text.strip()) if w.strip()]

    risk_preds, priority_scores = predict_tokens(tokens, MODEL_PATH)

    risky_spans = []
    for token, risk, score in zip(tokens, risk_preds, priority_scores):
        if risk == 1:
            risky_spans.append({
                "token": token,
                "risk": int(risk),
                "priority_score": round(float(score), 3)
            })

    max_priority = max(priority_scores) if priority_scores else 0.0

    if max_priority >= 0.8:
        route = "escalate"
    elif max_priority >= 0.4:
        route = "manual_review"
    else:
        route = "auto_reply"

    docs = retrieve_docs(text, top_k=top_k)
    suggestion = build_reply_suggestion(text, risky_spans, docs, route)

    return {
        "text": text,
        "tokens": tokens,
        "risk_predictions": risk_preds,
        "priority_scores": [round(float(x), 3) for x in priority_scores],
        "risky_spans": risky_spans,
        "max_priority": round(float(max_priority), 3),
        "route": route,
        "retrieved_docs": docs,
        "suggestion": suggestion
    }
