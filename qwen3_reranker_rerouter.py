# rerank_router.py
# Requires: fastapi, uvicorn, httpx

import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# ---------------------------
# Configuration
# ---------------------------

RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"       # your retriever endpoint (fetches 50 docs)
RERANKER_URL  = "http://127.0.0.1:8015/rerank"       # Qwen3 reranker endpoint
RETRIEVER_CANDIDATES = 50                            # how many docs to retrieve before reranking

# ---------------------------
# API Schema
# ---------------------------

class RouterRequest(BaseModel):
    queries: List[str]
    topk: int
    return_scores: bool = True

class RetrieverDoc(BaseModel):
    document: Dict[str, Any]
    score: float

class RouterResponse(BaseModel):
    result: List[List[RetrieverDoc]]

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(title="Retrieval → Reranker Router")

# ---------------------------
# API Logic
# ---------------------------

@app.post("/rerank", response_model=RouterResponse)
async def route_query(req: RouterRequest):
    # -----------------------------
    # 1. Call upstream retriever to get 50 candidates
    # -----------------------------
    retriever_payload = {
        "queries": req.queries,
        "topk": RETRIEVER_CANDIDATES,
        "return_scores": True
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        retriever_res = await client.post(RETRIEVER_URL, json=retriever_payload)
        retriever_data = retriever_res.json()

        all_results = retriever_data["result"]  

        # -----------------------------
        # 2. Prepare all reranker calls in parallel
        # -----------------------------
        async def rerank_query(q_idx: int, query: str, candidates: List[Dict]):
            """Rerank documents for a single query"""
            docs = [c["document"]["contents"] for c in candidates]
            
            reranker_payload = {
                "instruction": "Given a web search query, retrieve relevant passages that answer the query",
                "queries": [query] * len(docs),   # same query repeated
                "documents": docs
            }
            
            rerank_res = await client.post(RERANKER_URL, json=reranker_payload)
            rerank_data = rerank_res.json()
            rerank_scores = rerank_data["scores"]

            # Combine each document with rerank score
            enriched = []
            for doc_item, score in zip(candidates, rerank_scores):
                enriched.append({
                    "document": doc_item["document"],
                    "score": float(score)  # override original score
                })

            # Sort by rerank score and keep top-k
            enriched_sorted = sorted(enriched, key=lambda x: x["score"], reverse=True)
            topk_results = enriched_sorted[: req.topk]
            
            return topk_results

        # -----------------------------
        # 3. Make all reranker calls in parallel
        # -----------------------------
        tasks = [
            rerank_query(q_idx, query, all_results[q_idx])
            for q_idx, query in enumerate(req.queries)
        ]
        final_results = await asyncio.gather(*tasks)

    # -----------------------------
    # 4. Return identical format to retriever
    # -----------------------------
    return {"result": final_results}

# ---------------------------
# Start server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qwen3_reranker_rerouter:app", host="0.0.0.0", port=8025, reload=False)
