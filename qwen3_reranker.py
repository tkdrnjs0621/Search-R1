#!/usr/bin/env python3
import math
import torch
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


# =============================
# Model Setup
# =============================

MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
NUM_GPU = torch.cuda.device_count()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1,
    logprobs=20,
    allowed_token_ids=[true_token, false_token],
)

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=NUM_GPU if NUM_GPU > 0 else 1,
    max_model_len=8192,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.8,
)

# =============================
# Warmup (now runs exactly once)
# =============================
print("🔥 Warming up vLLM…")
llm.generate(
    [TokensPrompt(prompt_token_ids=[tokenizer.eos_token_id])],
    SamplingParams(max_tokens=1, temperature=0),
    use_tqdm=False,
)
print("✅ Warmup complete.\n")


# =============================
# Reranker Logic
# =============================

def format_instruction(instruction, query, doc):
    return [
        {
            "role": "system",
            "content": (
                'Judge whether the Document meets the requirements based on the Query '
                'and the Instruct provided. Answer only "yes" or "no".'
            )
        },
        {
            "role": "user",
            "content": (
                f"<Instruct>: {instruction}\n\n"
                f"<Query>: {query}\n\n"
                f"<Document>: {doc}"
            )
        }
    ]

def process_inputs(pairs, instruction, max_length=8192):
    messages = [format_instruction(instruction, q, d) for q, d in pairs]

    messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    messages = [
        m[: max_length - len(suffix_tokens)] + suffix_tokens for m in messages
    ]

    return [TokensPrompt(prompt_token_ids=m) for m in messages]

def compute_scores(outputs):
    scores = []
    for out in outputs:
        logits = out.outputs[0].logprobs[-1]

        t_logit = logits.get(true_token).logprob if true_token in logits else -10
        f_logit = logits.get(false_token).logprob if false_token in logits else -10

        t = math.exp(t_logit)
        f = math.exp(f_logit)
        scores.append(t / (t + f))
    return scores


# =============================
# FastAPI App
# =============================

app = FastAPI(title="Qwen3 Reranker API (vLLM)")

class RerankRequest(BaseModel):
    instruction: str
    queries: List[str]
    documents: List[str]

class RerankResponse(BaseModel):
    scores: List[float]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    pairs = list(zip(req.queries, req.documents))
    prompts = process_inputs(pairs, req.instruction)

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    scores = compute_scores(outputs)

    return RerankResponse(scores=scores)


# =============================
# Launch server on 8015 (NO DOUBLE LOAD)
# =============================

if __name__ == "__main__":
    print("🚀 Starting Qwen3 Reranker server at http://0.0.0.0:8015")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8015,
        workers=1,
        reload=False,      # <-- prevents double model load
    )
