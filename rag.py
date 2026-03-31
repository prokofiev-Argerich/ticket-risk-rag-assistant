from pathlib import Path

KB_DIR = Path("knowledge_base")


def retrieve_docs(query: str, top_k: int = 3):
    docs = []
    for path in KB_DIR.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        score = 0
        for keyword in ["退款", "投诉", "曝光", "仲裁", "客服", "订单", "发票"]:
            if keyword in query and keyword in content:
                score += 1
        docs.append({
            "title": path.name,
            "score": score,
            "content": content[:500]
        })

    docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    return docs[:top_k]
