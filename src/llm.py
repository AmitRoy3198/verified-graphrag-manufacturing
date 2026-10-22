"""The single place where we talk to the LLM.

- get_llm() builds a LangChain ChatOllama object (LangChain's wrapper around the
  local Ollama server at http://localhost:11434).
- log_llm_call() appends every prompt + response to results/llm_log.jsonl,
  so every LLM output in this project can be traced afterwards.
"""
import json
import time
from datetime import datetime, timezone

from langchain_ollama import ChatOllama

from src.config import MODEL, NUM_CTX, NUM_PREDICT, RESULTS, SEED, TEMPERATURE

LOG_FILE = RESULTS / "llm_log.jsonl"


def get_llm(json_mode: bool = False) -> ChatOllama:
    """temperature=0 + fixed seed + keep_alive=0 -> the same input gives the same output.

    keep_alive=0 unloads the model after every call. Phase 0 showed that when the
    model stays loaded, Ollama reuses cached computation for a repeated prompt and
    the answer can change (see results/phase0_cache_nondeterminism_log.jsonl).
    Reloading costs ~11 s per call but makes every call a clean, repeatable run.
    """
    return ChatOllama(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        keep_alive=0,
        format="json" if json_mode else None,  # Ollama can force valid JSON output
    )


def log_llm_call(phase: str, prompt_name: str, inputs: dict, output: str, seconds: float) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "prompt": prompt_name,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "inputs": inputs,
        "output": output,
        "seconds": round(seconds, 2),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_logged(chain, inputs: dict, phase: str, prompt_name: str) -> str:
    """Invoke a LangChain chain, log the call, return the text output."""
    start = time.perf_counter()
    result = chain.invoke(inputs)
    seconds = time.perf_counter() - start
    text = result.content if hasattr(result, "content") else str(result)
    log_llm_call(phase, prompt_name, inputs, text, seconds)
    return text
