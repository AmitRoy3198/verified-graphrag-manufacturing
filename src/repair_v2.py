"""Phase 3, v2: targeted + gated repair.

Why v2: in v1 (src/repair.py) the LLM rewrote whole entities and corrupted facts
that had nothing to do with the violation (results/phase3_v1_repair.json).

v2 changes three things:
1. TARGETED: only properties named in a SHACL violation can change. For every
   call we build a JSON schema that contains just those properties, so the
   model cannot even express a change to a locked fact.
2. MULTIPLE CHOICE: if a property has too many values (MaxCount), the schema
   is an enum of the existing values: the model picks one, it does not write.
3. GATED: the change is applied to a copy of the graph and kept only if the
   total number of SHACL violations goes down. Otherwise it is rolled back.

Outputs:
  results/phase3_v2_round<N>.ttl   graph after each round
  results/phase3_v2_repaired.ttl   final verified graph (used in Phase 4)
  results/phase3_v2_repair.json    per-round numbers and every repair attempt
Run:  python -m src.repair_v2
"""
import json
import time
from collections import defaultdict
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, create_model
from rdflib import Graph, URIRef

from src.config import DATA, DOCS, RESULTS, load_prompt
from src.extract_schema import Strict
from src.llm import get_llm, log_llm_call
from src.repair import FIELDS, MAX_DOCS, MAX_ROUNDS, code_of, node_class, provenance, to_value
from src.score import score
from src.validate import EX, MFG, validate_graph


# ---------- Which properties may change ----------
def prop_name(violation: dict) -> str:
    """SHACL result path -> our property name. 'mfg:powerKW' -> 'powerKW', '^mfg:feeds' -> 'fedBy'."""
    path = violation["path"]
    if not path:  # the SPARQL topology rule has no path: the machine must feed a buffer
        return "feeds"
    if path.startswith("^"):
        return "fedBy"
    return path.split(":")[-1]


def values_of(g: Graph, node, prop: str, cls) -> list[str]:
    """Current values of one property, as the LLM sees them (codes instead of IRIs)."""
    if prop == "fedBy":
        objs = list(g.subjects(MFG.feeds, node))
    else:
        objs = list(g.objects(node, FIELDS[cls][prop][0]))
    return sorted({code_of(o) if isinstance(o, URIRef) else str(o) for o in objs})


def plan_fix(g: Graph, node, cls, violations: list[dict]) -> dict:
    """{property: {"problems": [...], "choices": [...] or None}} for the violated properties."""
    plan = {}
    for v in violations:
        prop = prop_name(v)
        if prop not in FIELDS[cls]:
            continue
        entry = plan.setdefault(prop, {"problems": [], "choices": None})
        entry["problems"].append(v["message"])
        if v["component"] == "MaxCount":
            entry["choices"] = values_of(g, node, prop, cls)
    return plan


def fix_model(plan: dict):
    """Build a Pydantic model with one field per property to fix (enum when there are choices)."""
    fields = {}
    for prop, entry in plan.items():
        typ = Optional[Literal[tuple(entry["choices"])]] if entry["choices"] else Optional[str]
        fields[prop] = (typ, Field(None, description="null if the documents do not state it"))
    fields["delete_entity"] = (bool, Field(False, description="true only if the entity was created by mistake"))
    return create_model("Fix", __base__=Strict, **fields)


def local(node) -> str:
    """ex:buffer_B3 -> 'buffer_B3' (kind + code, unique)."""
    return str(node).rsplit("/", 1)[-1]


# ---------- Applying a fix ----------
def apply_fix(g: Graph, node, cls, plan: dict, answer: dict) -> None:
    """Set each planned property to exactly the answered value (null = no value)."""
    if answer.get("delete_entity"):
        g.remove((node, None, None))
        g.remove((None, None, node))
        return
    for prop in plan:
        p, kind, _ = FIELDS[cls][prop]
        if prop == "fedBy":
            g.remove((None, MFG.feeds, node))
        else:
            g.remove((node, p, None))
        value = answer.get(prop)
        if value is None or not str(value).strip():
            continue
        term = to_value(kind, str(value))
        if prop == "fedBy":
            g.add((term, MFG.feeds, node))
        else:
            g.add((node, p, term))


def gated_repair(g: Graph, node, violations: list[dict], ask) -> dict:
    """Ask for a fix, apply it to a copy, keep it only if total violations drop.

    `ask(inputs, model)` returns a dict answer or None. In the real run it calls
    the LLM; in the tests it is a fake, so the gate logic can be tested without an LLM.
    """
    cls = node_class(g, node)
    plan = plan_fix(g, node, cls, violations)
    before = len(validate_graph(g)[1])
    record = {"entity": code_of(node), "node": local(node), "class": cls.split("#")[-1],
              "plan": {k: {"choices": v["choices"], "current": values_of(g, node, k, cls)} for k, v in plan.items()}}
    if not plan:
        record.update(accepted=False, reason="no fixable property")
        return record

    answer = ask(build_inputs(g, node, cls, plan), fix_model(plan))
    record["answer"] = answer
    if answer is None:
        record.update(accepted=False, reason="no valid answer from LLM")
        return record

    trial = Graph()
    for t in g:
        trial.add(t)
    apply_fix(trial, node, cls, plan, answer)
    after = len(validate_graph(trial)[1])
    record.update(violations_before=before, violations_after=after, accepted=after < before)
    if after < before:
        g.remove((None, None, None))
        for t in trial:
            g.add(t)
    else:
        record["reason"] = "rolled back: violations did not drop"
    return record


# ---------- Prompt inputs and the LLM call ----------
SOURCES = {}  # filled in main(): entity IRI -> documents it was extracted from


def build_inputs(g, node, cls, plan) -> dict:
    locked = []
    for prop in FIELDS[cls]:
        if prop not in plan:
            locked += [f"- {prop}: {v}" for v in values_of(g, node, prop, cls)]
    to_fix = []
    for prop, entry in plan.items():
        line = f"- {prop} ({FIELDS[cls][prop][2]}). Problem: {' '.join(entry['problems'])}"
        if entry["choices"]:
            line += f" Choose one of: {', '.join(entry['choices'])}."
        else:
            current = values_of(g, node, prop, cls)
            line += f" Current value: {', '.join(current) if current else 'none'}."
        to_fix.append(line)
    docs = sorted(SOURCES.get(node, set()))[:MAX_DOCS]
    return {
        "entity": code_of(node),
        "cls": cls.split("#")[-1],
        "locked": "\n".join(locked) or "(none)",
        "to_fix": "\n".join(to_fix),
        "documents": "\n\n".join(f"[{d}]\n{(DOCS / (d + '.txt')).read_text(encoding='utf-8').strip()}"
                                 for d in docs) or "(no source document found)",
    }


def make_llm_ask(round_no: int):
    prompt = ChatPromptTemplate.from_messages([
        ("system", load_prompt("repair_v2_system")),
        ("human", load_prompt("repair_v2_user")),
    ])

    def ask(inputs: dict, model) -> dict | None:
        chain = prompt | get_llm().with_structured_output(model, method="json_schema", include_raw=True)
        start = time.perf_counter()
        result = chain.invoke(inputs)
        log_llm_call("phase3_v2", "repair_v2", {"round": round_no, **inputs},
                     result["raw"].content, time.perf_counter() - start)
        return None if result["parsed"] is None else result["parsed"].model_dump()

    return ask


def stats(g, gold) -> dict:
    conforms, violations = validate_graph(g)
    return {"conforms": conforms, "violations": len(violations),
            "entities_with_violations": len({v["focus"] for v in violations}),
            "triples": len(g), **score(g, gold)["overall"]}


def main() -> None:
    gold = Graph().parse(DATA / "ground_truth.ttl")
    g = Graph().parse(RESULTS / "phase2_extracted.ttl")
    SOURCES.update(provenance())

    rounds, repairs = [{"round": 0, **stats(g, gold)}], []
    print(rounds[-1], flush=True)
    for r in range(1, MAX_ROUNDS + 1):
        conforms, violations = validate_graph(g)
        if conforms:
            break
        grouped = defaultdict(list)
        for v in violations:
            grouped[v["focus"]].append(v)
        ask, accepted = make_llm_ask(r), 0
        for focus in sorted(grouped):
            node = EX[focus.split(":", 1)[1]]
            if node_class(g, node) is None:  # deleted earlier in this round
                continue
            # re-read this entity's violations: earlier fixes in this round may have changed them
            current = [v for v in validate_graph(g)[1] if v["focus"] == focus]
            if not current:
                continue
            start = time.perf_counter()
            rec = {"round": r, **gated_repair(g, node, current, ask), "seconds": round(time.perf_counter() - start, 1)}
            repairs.append(rec)
            accepted += rec["accepted"]
            print(f"round {r} {rec['entity']}: accepted={rec['accepted']} {rec.get('reason', '')}", flush=True)
        g.serialize(RESULTS / f"phase3_v2_round{r}.ttl", format="turtle")
        rounds.append({"round": r, "accepted_repairs": accepted, **stats(g, gold)})
        print(rounds[-1], flush=True)
        if accepted == 0:  # no progress: another round would ask the same questions
            break

    g.serialize(RESULTS / "phase3_v2_repaired.ttl", format="turtle")
    final = score(g, gold)
    gold_subjects = {local(s) for s in gold.subjects()}
    deleted = [x["node"] for x in repairs if x["accepted"] and (x.get("answer") or {}).get("delete_entity")]
    (RESULTS / "phase3_v2_repair.json").write_text(json.dumps({
        "rounds": rounds,
        "attempts": len(repairs),
        "accepted": sum(x["accepted"] for x in repairs),
        "deleted_entities": deleted,
        "deleted_entities_that_exist_in_ground_truth": [d for d in deleted if d in gold_subjects],
        "repairs": repairs,
        "final_per_predicate": final["per_predicate"],
        "final_missing": final["missing"],
        "final_wrong_or_extra": final["wrong_or_extra"],
        "remaining_violations": validate_graph(g)[1],
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
