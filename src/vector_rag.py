"""Baseline: plain vector RAG over the raw documents.

1. Each document is one chunk (they are short, ~50 words).
2. Hugging Face sentence-transformers (all-MiniLM-L6-v2) turns each chunk and the
   question into a 384-number vector.
3. LangChain's in-memory vector store returns the k chunks whose vectors are most
   similar (cosine) to the question vector.
4. The LLM answers from those chunks only.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DOCS, EMBED_MODEL

TOP_K = 4


def build_retriever():
    docs = [Document(page_content=p.read_text(encoding="utf-8"), metadata={"doc": p.stem})
            for p in sorted(DOCS.glob("*.txt"))]
    store = InMemoryVectorStore.from_documents(docs, HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    return store.as_retriever(search_kwargs={"k": TOP_K})


def retrieve(retriever, question: str) -> tuple[str, list[str]]:
    """Returns (context text for the prompt, names of the retrieved documents)."""
    hits = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['doc']}]\n{d.page_content.strip()}" for d in hits)
    return context, [d.metadata["doc"] for d in hits]
