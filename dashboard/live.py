"""Phase D3: the live pipeline behind the dashboard, without Streamlit (testable).

A new document goes through the same steps as the research pipeline:
extract (local LLM) -> RDF -> merge into the repaired graph -> SHACL check ->
targeted, gated repair (local LLM) -> saved.

Every live run writes ONLY to its own new folder dashboard_runs/<timestamp>_<kind>/
(input, LLM log, graphs, violations, repairs). results/ is never touched, so the
published numbers cannot change. LLM calls go to the local Ollama model only.
"""
import json
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdflib import Graph

import src.llm as llm
from dashboard import data
from src.config import ROOT
from src.extract import build_chain, extract_doc
from src.to_rdf import extraction_to_graph
from src.validate import EX

RUNS = ROOT / "dashboard_runs"
SAMPLES = data.DATA / "live_samples"
BASE = data.SOURCES["Repaired graph (after SHACL repair)"]


def ollama_version() -> str | None:
    """The local Ollama server's version, or None if it is not running."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2) as r:
            return json.load(r)["version"]
    except OSError:
        return None


def new_run_dir(kind: str) -> Path:
    d = RUNS / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{kind}"
    d.mkdir(parents=True, exist_ok=False)  # append-only: never reuse a folder
    return d


@contextmanager
def log_to(path: Path):
    """Send this run's LLM calls to the run folder instead of results/llm_log.jsonl."""
    old = llm.LOG_FILE
    llm.LOG_FILE = path
    try:
        yield
    finally:
        llm.LOG_FILE = old


# ---------------------------------------------------------------- the steps
def extract(text: str, run_dir: Path):
    """Step 1: the local LLM fills the extraction form for this document."""
    doc = run_dir / "document.txt"
    doc.write_text(text, encoding="utf-8")
    with log_to(run_dir / "llm_log.jsonl"):
        ext, info = extract_doc(build_chain(), doc)
    (run_dir / "extraction.json").write_text(ext.model_dump_json(indent=2), encoding="utf-8")
    return ext, info


def to_graph(ext, run_dir: Path) -> Graph:
    """Step 2: deterministic JSON -> RDF (no LLM)."""
    g = extraction_to_graph(ext)
    g.serialize(run_dir / "document_triples.ttl", format="turtle")
    return g


def merge(base: Graph, doc_graph: Graph) -> Graph:
    merged = Graph()
    for t in base:
        merged.add(t)
    for t in doc_graph:
        merged.add(t)
    for prefix, ns in base.namespaces():
        merged.bind(prefix, ns)
    return merged


def new_violations(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Step 3: violations present after merging that were not there before."""
    cols = ["entity", "rule", "check", "property", "value"]
    old = set(map(tuple, before[cols].astype(str).values))
    mask = [tuple(r) not in old for r in after[cols].astype(str).values]
    return after[mask].reset_index(drop=True)


def repair(merged: Graph, doc_graph: Graph, text: str, run_dir: Path, max_attempts: int = 5, ask=None) -> list:
    """Step 4: targeted, gated repair of the entities THIS document touched.

    `ask` is the LLM call (src.repair_v2.make_llm_ask); tests pass a fake. The
    live document is added to the source documents the model sees."""
    from src import repair_v2
    from src.repair import provenance

    if not repair_v2.SOURCES:
        repair_v2.SOURCES.update(provenance())
    viol = data.violations_df(merged)
    touched = {data.local(s) for s in doc_graph.subjects() if str(s).startswith(str(EX))}
    targets = sorted(touched & set(viol["entity"]))[:max_attempts]

    def ask_with_document(inputs, model):
        inputs = dict(inputs)
        known = "" if inputs["documents"] == "(no source document found)" else "\n\n" + inputs["documents"]
        inputs["documents"] = f"[new document]\n{text.strip()}{known}"
        return (ask or repair_v2.make_llm_ask(round_no=1))(inputs, model)

    records = []
    with log_to(run_dir / "llm_log.jsonl"):
        for name in targets:
            current = [v for v in data.validate_graph(merged)[1] if v["focus"] == f"ex:{name}"]
            if current:
                # per-property gate: a good fix can no longer carry a bad one (found in a live run)
                records.append(repair_v2.gated_repair(merged, EX[name], current, ask_with_document,
                                                      per_property=True))
    (run_dir / "repairs.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return records


def save(run_dir: Path, merged: Graph, before: pd.DataFrame, after: pd.DataFrame, summary: dict) -> None:
    merged.serialize(run_dir / "graph_after.ttl", format="turtle")
    after.to_json(run_dir / "violations_after.json", orient="records", indent=2)
    (run_dir / "summary.json").write_text(json.dumps(
        {**summary, "violations_base": len(before), "violations_after": len(after),
         "base_graph": str(BASE.relative_to(ROOT))}, indent=2, default=str), encoding="utf-8")


def live_runs() -> list[Path]:
    """Finished pipeline runs, newest first (they can be opened as a graph on every page)."""
    return sorted((p for p in RUNS.glob("*_document") if (p / "graph_after.ttl").exists()), reverse=True)


# ---------------------------------------------------------------- live question
def ask_question(question: str, methods: list[str], run_dir: Path) -> list[dict]:
    """Answer one question with the chosen methods, logging into the run folder."""
    from src import graph_rag, vector_rag
    from src.qa import answer_chain, ask

    graphs = {"GraphRAG · repaired graph": BASE, "GraphRAG · perfect graph": data.DATA / "ground_truth.ttl"}
    out = []
    with log_to(run_dir / "llm_log.jsonl"):
        for m in methods:
            if m == "Vector RAG":
                context, docs = vector_rag.retrieve(vector_rag.build_retriever(), question)
                info = {"retrieved": docs}
                chain = answer_chain("qa_vector_system")
            else:
                context, info = graph_rag.retrieve(data.load_graph(graphs[m]), question)
                chain = answer_chain("qa_graph_system")
            out.append({"method": m, **ask(chain, question, context, f"live_{m}"), "info": info,
                        "context": context})
    (run_dir / "answers.json").write_text(json.dumps({"question": question, "answers": out}, indent=2,
                                                     ensure_ascii=False, default=str), encoding="utf-8")
    return out
