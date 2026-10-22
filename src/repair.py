"""Phase 3: verify with SHACL, repair with the LLM, repeat (max 3 rounds).

One round:
  1. validate the graph (pySHACL) -> violations
  2. group violations by entity (focus node)
  3. for each entity ask the LLM for its corrected, complete fact list (or deletion),
     giving it the violation messages and the documents the entity came from
  4. replace the entity's facts in the graph
Stops early when the graph conforms.

Outputs:
  results/phase3_round<N>.ttl      graph after each round
  results/phase3_repair.json       violations + P/R/F1 per round, and every repair
Run:  python -m src.repair
"""
import json
import time
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
from rdflib import Graph, Literal, RDF, RDFS, URIRef

from src.config import DATA, DOCS, RESULTS, load_prompt
from src.extract_schema import Extraction, Strict
from src.llm import get_llm, log_llm_call
from src.score import score
from src.to_rdf import dec, date_lit, extraction_to_graph, iri, whole
from src.validate import EX, MFG, validate_graph

MAX_ROUNDS = 3
MAX_DOCS = 8  # source documents per repair prompt (keeps the prompt inside the context window)


# ---------- What the LLM returns ----------
class Fact(Strict):
    property: str = Field(description="One of the allowed property names")
    value: str = Field(description="The value, as text")


class Repair(Strict):
    delete_entity: bool = Field(description="True only if this entity should not exist")
    facts: list[Fact] = Field(description="The complete corrected list of facts")


# ---------- Allowed properties per class: name -> (RDF property, value kind, description) ----------
# value kind: text | decimal | whole | hour | date | <kind of entity referred to>
FIELDS = {
    MFG.Machine: {
        "label": (RDFS.label, "text", "full name, e.g. 'Press M7' (one)"),
        "machineType": (MFG.machineType, "text", "Press, Washer, Assembly, Inspection or Packing (one)"),
        "cycleTimeS": (MFG.cycleTimeS, "decimal", "cycle time in seconds (one)"),
        "powerKW": (MFG.powerKW, "decimal", "rated power in kW (one)"),
        "mtbfH": (MFG.mtbfH, "decimal", "mean time between failures in hours (one)"),
        "mttrH": (MFG.mttrH, "decimal", "mean time to repair in hours (one)"),
        "partOfLine": (MFG.partOfLine, "line", "name of its production line (one)"),
        "feeds": (MFG.feeds, "buffer", "code of the buffer it feeds (none for the last machine)"),
    },
    MFG.Buffer: {
        "label": (RDFS.label, "text", "full name, e.g. 'Buffer B9' (one)"),
        "capacity": (MFG.capacity, "whole", "capacity in parts (one)"),
        "feeds": (MFG.feeds, "machine", "code of the machine it feeds (one)"),
        "fedBy": (None, "machine", "code of the machine that feeds it (one)"),  # stored as machine feeds buffer
    },
    MFG.Failure: {
        "onMachine": (MFG.onMachine, "machine", "code of the machine that failed (one)"),
        "failureMode": (MFG.failureMode, "text", "failure mode (one)"),
        "downtimeH": (MFG.downtimeH, "decimal", "production downtime in hours (one)"),
        "date": (MFG.date, "date", "YYYY-MM-DD (one)"),
        "duringShift": (MFG.duringShift, "shift", "Early, Late or Night (at most one)"),
    },
    MFG.MaintenanceAction: {
        "onMachine": (MFG.onMachine, "machine", "code of the machine worked on (one)"),
        "actionType": (MFG.actionType, "text", "Preventive or Corrective (one)"),
        "date": (MFG.date, "date", "YYYY-MM-DD (one)"),
        "durationH": (MFG.durationH, "decimal", "work duration in hours (one)"),
        "addresses": (MFG.addresses, "failure", "incident code it repaired (at most one)"),
    },
    MFG.Shift: {
        "label": (RDFS.label, "text", "full name, e.g. 'Early shift' (one)"),
        "startHour": (MFG.startHour, "hour", "start hour of day, 0-23 (one)"),
    },
    MFG.ProductionLine: {
        "label": (RDFS.label, "text", "name of the line (one)"),
        "finalMachine": (MFG.finalMachine, "machine", "code of the last machine (one)"),
    },
}


def node_class(g: Graph, node):
    for t in g.objects(node, RDF.type):
        if t in FIELDS:
            return t
    return None


def code_of(node) -> str:
    """ex:buffer_B3 -> 'B3' (what the LLM sees instead of the full IRI)."""
    return str(node).rsplit("/", 1)[-1].split("_", 1)[-1]


def to_value(kind: str, text: str):
    """LLM text -> RDF term. A value that cannot be converted stays plain text, so SHACL can flag it."""
    text = text.strip()
    try:
        if kind == "decimal":
            return dec(float(text))
        if kind == "whole":
            return whole(float(text))
        if kind == "hour":
            return Literal(int(float(text)))
    except ValueError:
        return Literal(text)
    if kind == "date":
        return date_lit(text)
    if kind == "text":
        return Literal(text)
    return iri(kind, text)  # a reference to another entity


# ---------- Reading and writing one entity ----------
def current_facts(g: Graph, node, cls) -> list[tuple[str, str]]:
    by_prop = {p: name for name, (p, _k, _d) in FIELDS[cls].items() if p is not None}
    facts = []
    for p, o in g.predicate_objects(node):
        if p in by_prop:
            facts.append((by_prop[p], code_of(o) if isinstance(o, URIRef) else str(o)))
    if cls == MFG.Buffer:
        facts += [("fedBy", code_of(m)) for m in g.subjects(MFG.feeds, node)]
    return sorted(facts)


def apply_repair(g: Graph, node, cls, repair: Repair) -> list[str]:
    """Replace the entity's facts with the repaired ones. Returns notes about ignored facts."""
    if repair.delete_entity:
        g.remove((node, None, None))
        g.remove((None, None, node))
        return ["entity deleted"]
    for p, _o in list(g.predicate_objects(node)):
        if p != RDF.type:
            g.remove((node, p, None))
    if cls == MFG.Buffer:
        g.remove((None, MFG.feeds, node))
    notes = []
    for fact in repair.facts:
        if fact.property not in FIELDS[cls]:
            notes.append(f"ignored unknown property '{fact.property}'")
            continue
        p, kind, _ = FIELDS[cls][fact.property]
        value = to_value(kind, fact.value)
        if p is None:  # fedBy: machine feeds this buffer
            g.add((value, MFG.feeds, node))
        else:
            g.add((node, p, value))
    return notes


# ---------- Which documents an entity came from (provenance from Phase 2) ----------
def provenance() -> dict:
    sources = defaultdict(set)
    for path in sorted((RESULTS / "phase2_extractions").glob("*.json")):
        g = extraction_to_graph(Extraction.model_validate_json(path.read_text(encoding="utf-8")))
        for s, _p, o in g:
            sources[s].add(path.stem)
            if isinstance(o, URIRef) and str(o).startswith(str(EX)):
                sources[o].add(path.stem)
    return sources


# ---------- The loop ----------
def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("repair_system")),
        ("human", load_prompt("repair_user")),
    ])
    return prompt | get_llm().with_structured_output(Repair, method="json_schema", include_raw=True)


def repair_entity(chain, g, node, violations, sources, round_no) -> dict:
    cls = node_class(g, node)
    name = code_of(node)
    docs = sorted(sources.get(node, set()))[:MAX_DOCS]
    inputs = {
        "entity": name,
        "cls": cls.split("#")[-1],
        "allowed": "\n".join(f"- {k}: {d}" for k, (_p, _kind, d) in FIELDS[cls].items()),
        "facts": "\n".join(f"- {k}: {v}" for k, v in current_facts(g, node, cls)) or "(none)",
        "violations": "\n".join(f"- {v['path'] or 'topology'}: {v['message']}"
                                + (f" (value: {v['value']})" if v["value"] else "") for v in violations),
        "documents": "\n\n".join(f"[{d}]\n{(DOCS / (d + '.txt')).read_text(encoding='utf-8').strip()}"
                                 for d in docs) or "(no source document found)",
    }
    start = time.perf_counter()
    result = chain.invoke(inputs)
    seconds = time.perf_counter() - start
    log_llm_call("phase3", "repair", {"round": round_no, **inputs}, result["raw"].content, seconds)

    record = {"round": round_no, "entity": name, "violations": [v["message"] for v in violations],
              "facts_before": current_facts(g, node, cls), "seconds": round(seconds, 1)}
    if result["parsed"] is None:
        record["error"] = str(result["parsing_error"])  # leave the entity unchanged
        return record
    record["notes"] = apply_repair(g, node, cls, result["parsed"])
    record["facts_after"] = [] if result["parsed"].delete_entity else current_facts(g, node, cls)
    return record


def stats(g, gold) -> dict:
    conforms, violations = validate_graph(g)
    return {"conforms": conforms, "violations": len(violations),
            "entities_with_violations": len({v["focus"] for v in violations}),
            "triples": len(g), **score(g, gold)["overall"]}


def main() -> None:
    gold = Graph().parse(DATA / "ground_truth.ttl")
    g = Graph().parse(RESULTS / "phase2_extracted.ttl")
    sources, chain = provenance(), build_chain()

    rounds = [{"round": 0, **stats(g, gold)}]
    repairs = []
    print(rounds[-1], flush=True)
    for r in range(1, MAX_ROUNDS + 1):
        conforms, violations = validate_graph(g)
        if conforms:
            break
        grouped = defaultdict(list)
        for v in violations:
            grouped[v["focus"]].append(v)
        for focus, vs in sorted(grouped.items()):
            node = EX[focus.split(":", 1)[1]]
            if node_class(g, node) is None:  # deleted earlier in this round
                continue
            rec = repair_entity(chain, g, node, vs, sources, r)
            repairs.append(rec)
            print(f"round {r} {rec['entity']}: {rec.get('notes', rec.get('error'))} ({rec['seconds']} s)", flush=True)
        g.serialize(RESULTS / f"phase3_round{r}.ttl", format="turtle")
        rounds.append({"round": r, **stats(g, gold)})
        print(rounds[-1], flush=True)

    final = score(g, gold)
    (RESULTS / "phase3_repair.json").write_text(json.dumps({
        "rounds": rounds, "repairs": repairs,
        "final_per_predicate": final["per_predicate"],
        "final_missing": final["missing"], "final_wrong_or_extra": final["wrong_or_extra"],
        "remaining_violations": validate_graph(g)[1],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
