"""Phase 2: LLM extraction, one document at a time, then merge and score.

For each document:  prompt (LangChain)  ->  Ollama with JSON-schema constrained
output  ->  Pydantic Extraction  ->  RDF (src/to_rdf.py).
The union of all per-document graphs is the extracted knowledge graph.

Outputs:
  results/phase2_extractions/<doc>.json   parsed JSON per document
  results/phase2_extracted.ttl            merged graph
  results/phase2_scores.json              precision / recall / F1 vs ground truth
Run:  python -m src.extract            (all docs)
      python -m src.extract 02_datasheet_P1   (one doc, for a quick check)
"""
import json
import sys
import time

from langchain_core.prompts import ChatPromptTemplate
from rdflib import Graph

from src.config import DATA, DOCS, RESULTS, load_prompt
from src.extract_schema import Extraction
from src.llm import get_llm, log_llm_call
from src.score import score
from src.to_rdf import extraction_to_graph

OUT_DIR = RESULTS / "phase2_extractions"


def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("extract_system")),
        ("human", load_prompt("extract_user")),
    ])
    # with_structured_output: LangChain sends the Pydantic JSON schema to Ollama,
    # which constrains decoding to it, then parses the reply into an Extraction.
    # include_raw=True keeps the raw text so we can log it.
    llm = get_llm().with_structured_output(Extraction, method="json_schema", include_raw=True)
    return prompt | llm


def extract_doc(chain, path) -> tuple[Extraction, dict]:
    text = path.read_text(encoding="utf-8")
    start = time.perf_counter()
    result = chain.invoke({"document": text})
    seconds = time.perf_counter() - start
    raw = result["raw"].content
    log_llm_call("phase2", "extract", {"doc": path.name}, raw, seconds)
    parsed = result["parsed"]
    info = {"doc": path.name, "seconds": round(seconds, 1), "parse_error": None}
    if parsed is None:  # invalid JSON / schema mismatch: count as an empty extraction
        info["parse_error"] = str(result["parsing_error"])
        parsed = Extraction()
    return parsed, info


def main(only: str | None = None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    docs = sorted(DOCS.glob("*.txt"))
    if only:
        docs = [d for d in docs if d.stem == only]
    chain = build_chain()

    merged, runs = Graph(), []
    for path in docs:
        ext, info = extract_doc(chain, path)
        g = extraction_to_graph(ext)
        info["triples"] = len(g)
        runs.append(info)
        (OUT_DIR / f"{path.stem}.json").write_text(ext.model_dump_json(indent=2), encoding="utf-8")
        merged += g
        print(f"{path.name}: {len(g)} triples, {info['seconds']} s", flush=True)

    if only:  # quick check: print the JSON, do not overwrite the full results
        print((OUT_DIR / f"{only}.json").read_text(encoding="utf-8"))
        return

    merged.bind("ex", "http://example.org/line/")
    merged.bind("mfg", "http://example.org/mfg#")
    merged.serialize(RESULTS / "phase2_extracted.ttl", format="turtle")
    gold = Graph().parse(DATA / "ground_truth.ttl")
    result = {"runs": runs, "total_seconds": round(sum(r["seconds"] for r in runs), 1),
              **score(merged, gold)}
    (RESULTS / "phase2_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
