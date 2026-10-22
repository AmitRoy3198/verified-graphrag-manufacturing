"""Central settings. Every script imports from here, so one change applies everywhere."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
SCHEMA = ROOT / "schema"
DATA = ROOT / "data"
DOCS = DATA / "docs"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Keep the Hugging Face model cache inside the project (not on C:).
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

# LLM settings: one local model, deterministic decoding.
MODEL = "qwen2.5:3b-instruct"
TEMPERATURE = 0
SEED = 42
NUM_CTX = 4096  # context window in tokens; small to save RAM on an 8 GB laptop
NUM_PREDICT = 1500  # max output tokens; v1 showed the model can loop forever without a cap

# Embedding model for the vector-RAG baseline (Phase 4).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_prompt(name: str) -> str:
    """All prompts live in prompts/*.txt so they are versioned and easy to inspect."""
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
