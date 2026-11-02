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
from rdflib import RDF, RDFS, Graph, URIRef

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


# ---------------------------------------------------------------- reading a run back (the operator view)
# Plain-language names for the properties, so the operator does not read RDF.
PROP_LABEL = {
    "label": "name", "machineType": "type", "cycleTimeS": "cycle time (s)", "powerKW": "rated power (kW)",
    "mtbfH": "MTBF (h)", "mttrH": "MTTR (h)", "partOfLine": "belongs to line", "feeds": "feeds",
    "capacity": "capacity (parts)", "onMachine": "machine", "failureMode": "failure mode",
    "downtimeH": "downtime (h)", "date": "date", "duringShift": "shift", "actionType": "action type",
    "durationH": "duration (h)", "addresses": "fixes incident", "finalMachine": "last machine",
}
AGENTS = {"extract": "Reader agent", "repair_v2": "Repair agent"}


def prop_of(predicate) -> str:
    """mfg#powerKW / rdfs#label -> 'powerKW' / 'label'."""
    return str(predicate).split("#")[-1]


def name_of(g: Graph, node) -> str:
    """A node as the operator knows it: its label if it has one, else its code."""
    if not isinstance(node, URIRef):
        return str(node)
    labels = sorted(map(str, g.objects(node, RDFS.label)))
    return labels[0] if labels else data.local(node).split("_", 1)[-1]


def touched(doc_graph: Graph) -> set[str]:
    """Entities the document mentions, e.g. {'machine_W1', 'line_A'}."""
    return {data.local(s) for s in doc_graph.subjects() if str(s).startswith(str(EX))}


def fact_rows(base: Graph, doc_graph: Graph, new_viol: pd.DataFrame) -> list[dict]:
    """Every fact the document gave, sorted into new / already known / conflict.

    conflict = the fact is part of a new "too many values" violation (MaxCount),
    i.e. our records already hold a different value for a one-value property."""
    clash = {(r["entity"], r["property"].split(":")[-1]) for _, r in new_viol.iterrows() if r["check"] == "MaxCount"}
    names = Graph()
    for t in base:
        names.add(t)
    for t in doc_graph:
        names.add(t)
    rows = []
    for s, p, o in doc_graph:
        if p == RDF.type:
            continue
        prop = prop_of(p)
        if (data.local(s), prop) in clash:
            status = "conflict"
        elif (s, p, o) in base:
            status = "already known"
        else:
            status = "new"
        value = name_of(names, o)
        earlier = sorted(name_of(names, x) for x in base.objects(s, p) if x != o)
        if value in earlier:  # same name, different entity (e.g. two "Line A" nodes): show the codes
            value, earlier = f"{value} ({data.local(o)})", [f"{name_of(names, x)} ({data.local(x)})"
                                                            for x in base.objects(s, p) if x != o]
        rows.append({"status": status, "entity": data.local(s), "what": name_of(names, s),
                     "property": PROP_LABEL.get(prop, prop), "value": value,
                     "our records say": ", ".join(earlier) if status == "conflict" else ""})
    order = {"conflict": 0, "new": 1, "already known": 2}
    return sorted(rows, key=lambda r: (order[r["status"]], r["what"], r["property"]))


def agent_calls(run_dir: Path) -> list[dict]:
    """The LLM calls of this run, each with exactly the messages the model received."""
    from langchain_core.prompts import ChatPromptTemplate
    from src.config import load_prompt

    path = run_dir / "llm_log.jsonl"
    if not path.exists():
        return []
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        name = rec["prompt"]
        if name == "extract":
            inputs = {"document": (run_dir / "document.txt").read_text(encoding="utf-8")}
            task = "Read the document and fill the extraction form"
        else:
            inputs = {k: v for k, v in rec["inputs"].items() if k != "round"}
            props = [ln.lstrip("- ").split(" (")[0] for ln in inputs["to_fix"].splitlines()]
            task = f"Fix {inputs['entity']}: " + ", ".join(PROP_LABEL.get(x, x) for x in props)
        stem = "extract" if name == "extract" else name
        messages = ChatPromptTemplate.from_messages([
            ("system", load_prompt(f"{stem}_system")), ("human", load_prompt(f"{stem}_user"))
        ]).format_messages(**inputs)
        calls.append({"agent": AGENTS.get(name, name), "task": task, "seconds": rec["seconds"],
                      "model": rec["model"], "temperature": rec["temperature"], "seed": rec["seed"],
                      "system": messages[0].content, "user": messages[1].content, "output": rec["output"]})
    return calls


def open_questions(g: Graph, doc_graph: Graph) -> list[dict]:
    """Conflicts the AI left open: a one-value property of an entity this document
    touched still holds several values. The operator chooses."""
    from src import repair_v2
    from src.repair import node_class

    out = []
    for name in sorted(touched(doc_graph)):
        node = EX[name]
        viol = [v for v in data.validate_graph(g)[1] if v["focus"] == f"ex:{name}" and v["component"] == "MaxCount"]
        cls = node_class(g, node)
        if not viol or cls is None:
            continue
        for prop, entry in repair_v2.plan_fix(g, node, cls, viol).items():
            if not entry["choices"]:
                continue
            p = repair_v2.FIELDS[cls][prop][0]
            objs = doc_graph.subjects(data.MFG.feeds, node) if p is None else doc_graph.objects(node, p)  # fedBy
            in_doc = {repair_v2.code_of(o) if isinstance(o, URIRef) else str(o) for o in objs}
            out.append({"entity": name, "what": name_of(g, node), "property": prop,
                        "label": PROP_LABEL.get(prop, prop), "problem": entry["problems"][0],
                        "choices": [{"value": c, "source": "this document" if c in in_doc else "our records"}
                                    for c in entry["choices"]]})
    return out


def decide(run_dir: Path, entity: str, prop: str, value: str | None) -> dict:
    """The operator's choice: keep exactly `value` for this property. Written only
    to this run's folder; the AI's graph is kept as graph_before_decisions.ttl."""
    from src import repair_v2
    from src.repair import node_class

    path = run_dir / "graph_after.ttl"
    keep = run_dir / "graph_before_decisions.ttl"
    if not keep.exists():
        keep.write_bytes(path.read_bytes())
    g = data.load_graph(path)
    node = EX[entity]
    cls = node_class(g, node)
    before = len(data.validate_graph(g)[1])
    plan = {prop: {"choices": repair_v2.values_of(g, node, prop, cls)}}
    repair_v2.apply_fix(g, node, cls, plan, {prop: value})
    after_v = data.violations_df(g)
    g.serialize(path, format="turtle")
    after_v.to_json(run_dir / "violations_after.json", orient="records", indent=2)
    rec = {"time": datetime.now(timezone.utc).isoformat(), "by": "operator", "entity": entity, "property": prop,
           "choices": plan[prop]["choices"], "chosen": value,
           "violations_before": before, "violations_after": len(after_v)}
    log = run_dir / "decisions.json"
    done = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    log.write_text(json.dumps(done + [rec], indent=2), encoding="utf-8")
    return rec


def decisions(run_dir: Path) -> list[dict]:
    log = run_dir / "decisions.json"
    return json.loads(log.read_text(encoding="utf-8")) if log.exists() else []


def values_now(g: Graph, entity: str, path: str) -> list[str]:
    """Current values behind one SHACL result path ('mfg:powerKW', '^mfg:feeds', 'rdfs:label')."""
    node, name = EX[entity], path.split(":")[-1]
    if not path:
        return []
    if path.startswith("^"):
        objs = g.subjects(data.MFG[name], node)
    elif path == "rdfs:label":
        objs = g.objects(node, RDFS.label)
    else:
        objs = g.objects(node, data.MFG[name])
    return sorted(name_of(g, o) for o in objs)
