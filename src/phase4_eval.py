"""Phase 4: answer the 25 questions under four conditions and score them.

Conditions
  vector_rag            plain vector RAG over the raw documents (baseline)
  graphrag_unrepaired   GraphRAG over the Phase 2 graph (before SHACL repair)
  graphrag_repaired     GraphRAG over the Phase 3 v2 graph (the "verified" graph)
  graphrag_gold         GraphRAG over the hand-written ground truth (upper bound)

Outputs
  results/phase4_answers_<condition>.json   every question, context size, answer, scores
  results/phase4_scores.json                accuracy overall and per question type
Run:  python -m src.phase4_eval                      (all conditions)
      python -m src.phase4_eval vector_rag           (one condition)
      python -m src.phase4_eval --summary            (only rebuild phase4_scores.json)
"""
import json
from math import comb
import sys

from rdflib import Graph

from src import graph_rag, vector_rag
from src.config import DATA, RESULTS
from src.qa import answer_chain, ask, exact_match, lenient_match, load_questions

GRAPHS = {
    "graphrag_unrepaired": RESULTS / "phase2_extracted.ttl",
    "graphrag_repaired": RESULTS / "phase3_v2_repaired.ttl",
    "graphrag_gold": DATA / "ground_truth.ttl",
}
CONDITIONS = ["vector_rag", *GRAPHS]
TYPES = ["1-hop", "multi-hop", "aggregation"]


def run(condition: str) -> None:
    questions = load_questions()
    if condition == "vector_rag":
        retriever = vector_rag.build_retriever()
        chain = answer_chain("qa_vector_system")
    else:
        graph = Graph().parse(GRAPHS[condition])
        chain = answer_chain("qa_graph_system")

    rows = []
    for q in questions:
        if condition == "vector_rag":
            context, docs = vector_rag.retrieve(retriever, q["question"])
            info = {"retrieved_docs": docs,
                    "doc_recall": round(len(set(docs) & set(q["docs"])) / len(q["docs"]), 3)}
        else:
            context, info = graph_rag.retrieve(graph, q["question"])
        out = ask(chain, q["question"], context, condition)
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "expected": q["answer"],
                     **out, "exact": exact_match(out["answer"], q), "lenient": lenient_match(out["answer"], q),
                     "context_chars": len(context), **info})
        r = rows[-1]
        print(f"{condition} {q['id']}: {r['answer']!r} (expected {q['answer']}) exact={r['exact']} "
              f"lenient={r['lenient']} {r['seconds']} s", flush=True)
    (RESULTS / f"phase4_answers_{condition}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def accuracy(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None


def summary() -> dict:
    out = {}
    for c in CONDITIONS:
        path = RESULTS / f"phase4_answers_{c}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        out[c] = {
            "n": len(rows),
            "exact": accuracy(rows, "exact"),
            "lenient": accuracy(rows, "lenient"),
            "per_type": {t: {"n": len([r for r in rows if r["type"] == t]),
                             "exact": accuracy([r for r in rows if r["type"] == t], "exact"),
                             "lenient": accuracy([r for r in rows if r["type"] == t], "lenient")} for t in TYPES},
            "answered_unknown": sum(r["answer"].strip().lower() == "unknown" for r in rows),
            "parse_errors": sum("error" in r for r in rows),
        }
        if c == "vector_rag":
            out[c]["mean_doc_recall"] = round(sum(r["doc_recall"] for r in rows) / len(rows), 3)
            out[c]["questions_with_all_docs_retrieved"] = sum(r["doc_recall"] == 1.0 for r in rows)
    out["paired_comparisons_lenient"] = paired(PAIRS)
    (RESULTS / "phase4_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


PAIRS = [("vector_rag", "graphrag_repaired"), ("graphrag_unrepaired", "graphrag_repaired"),
         ("vector_rag", "graphrag_gold")]


def paired(pairs) -> dict:
    """Per pair: on how many questions only A or only B was right (lenient), plus a
    two-sided sign test on those disagreeing questions. With 25 questions, small
    accuracy differences are usually not significant."""
    res = {}
    for a, b in pairs:
        pa, pb = (RESULTS / f"phase4_answers_{a}.json"), (RESULTS / f"phase4_answers_{b}.json")
        if not (pa.exists() and pb.exists()):
            continue
        ra = {r["id"]: r["lenient"] for r in json.loads(pa.read_text(encoding="utf-8"))}
        rb = {r["id"]: r["lenient"] for r in json.loads(pb.read_text(encoding="utf-8"))}
        only_a = sum(ra[q] and not rb[q] for q in ra)
        only_b = sum(rb[q] and not ra[q] for q in ra)
        n, k = only_a + only_b, min(only_a, only_b)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        res[f"{a} vs {b}"] = {"both_right": sum(ra[q] and rb[q] for q in ra), f"only_{a}": only_a,
                              f"only_{b}": only_b, "both_wrong": sum(not ra[q] and not rb[q] for q in ra),
                              "sign_test_p": round(p, 3)}
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    if args != ["--summary"]:
        for c in args or CONDITIONS:
            run(c)
    print(json.dumps(summary(), indent=2))
