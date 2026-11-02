"""Data layer for the dashboard. No Streamlit here, so it can be unit-tested.

Rule (borrowed from DES v2): no LLM computes a number that reaches the operator.
Every number below comes either from a results/ file or from the knowledge
graph through SPARQL / plain Python over the triples.
"""
import json
from collections import Counter, defaultdict

import pandas as pd
from rdflib import Graph, Literal, RDF, RDFS, URIRef

from src.config import DATA, DOCS, RESULTS
from src.repair import code_of
from src.score import score
from src.validate import EX, MFG, validate_graph

# The three versions of the graph an operator can switch between.
SOURCES = {
    "Repaired graph (after SHACL repair)": RESULTS / "phase3_v2_repaired.ttl",
    "Extracted graph (before repair)": RESULTS / "phase2_extracted.ttl",
    "Ground truth (hand-written)": DATA / "ground_truth.ttl",
}

PREFIXES = """PREFIX ex: <http://example.org/line/>
PREFIX mfg: <http://example.org/mfg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

# The SPARQL behind each chart. The pages show these in an expander.
QUERIES = {
    "downtime_by_machine": """SELECT ?machine (COUNT(DISTINCT ?f) AS ?failures) (SUM(COALESCE(?d, 0)) AS ?downtime_h)
WHERE {
  ?f a mfg:Failure ; mfg:onMachine ?machine .
  OPTIONAL { ?f mfg:downtimeH ?d }
}
GROUP BY ?machine""",
    "failures_by_shift": """SELECT ?shift (COUNT(DISTINCT ?f) AS ?failures)
WHERE {
  ?f a mfg:Failure .
  OPTIONAL { ?f mfg:duringShift ?s }
  BIND (COALESCE(?s, "unknown") AS ?shift)
}
GROUP BY ?shift""",
    "actions_by_type": """SELECT ?type (COUNT(DISTINCT ?a) AS ?actions) (SUM(COALESCE(?h, 0)) AS ?hours)
WHERE {
  ?a a mfg:MaintenanceAction .
  OPTIONAL { ?a mfg:actionType ?t }
  OPTIONAL { ?a mfg:durationH ?h }
  BIND (COALESCE(STR(?t), "unknown") AS ?type)
}
GROUP BY ?type""",
}


def local(node) -> str:
    """ex:machine_P2 -> 'machine_P2'."""
    return str(node).rsplit("/", 1)[-1]


def num(o):
    """A literal as float, or None if it is not a number (e.g. the text 'bytes')."""
    try:
        return float(o)
    except (TypeError, ValueError):
        return None


def load_graph(path) -> Graph:
    return Graph().parse(path)


def query(g: Graph, name: str) -> pd.DataFrame:
    rows = g.query(PREFIXES + QUERIES[name])
    cols = [str(v) for v in rows.vars]
    data = [[code_of(x) if isinstance(x, URIRef) else (num(x) if num(x) is not None else str(x))
             for x in row] for row in rows]
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------- line layout
def _walk(g: Graph) -> list:
    """Machines reached by following the flow from the start machine
    (the machine no buffer feeds): machine -> buffer -> machine -> ..."""
    machines = set(g.subjects(RDF.type, MFG.Machine))
    fed = {o for s in g.subjects(RDF.type, MFG.Buffer) for o in g.objects(s, MFG.feeds)}
    # start = the machine no buffer feeds that has the longest downstream walk
    best = []
    for start in sorted(m for m in machines if m not in fed):
        order, current = [], start
        while current is not None and current not in order:
            order.append(current)
            buffers = sorted(b for b in g.objects(current, MFG.feeds) if (b, RDF.type, MFG.Buffer) in g)
            # first machine reachable through a buffer that does not loop back
            nxt = [m for b in buffers for m in sorted(g.objects(b, MFG.feeds)) if m in machines and m not in order]
            current = nxt[0] if nxt else None
        best = order if len(order) > len(best) else best
    return best


def line_order(g: Graph) -> list[str]:
    """Machine codes in flow order; machines not on the flow are appended, sorted."""
    walk = _walk(g)
    rest = sorted(set(g.subjects(RDF.type, MFG.Machine)) - set(walk))
    return [code_of(m) for m in walk + rest]


def flow_sequence(g: Graph) -> tuple[list[str], list[str]]:
    """The line as one sequence of entity names (machine, buffer, machine, ...),
    plus every machine/buffer that is not on that sequence (duplicates, inventions)."""
    machines = _walk(g)
    seq = []
    if not machines:
        everything = {local(s) for t in (MFG.Machine, MFG.Buffer) for s in g.subjects(RDF.type, t)}
        return seq, sorted(everything)
    for a, b in zip(machines, machines[1:] + [None]):
        seq.append(local(a))
        if b is None:
            break
        link = sorted(buf for buf in g.objects(a, MFG.feeds)
                      if (buf, MFG.feeds, b) in g and local(buf) not in seq)  # each node drawn once
        if link:
            seq.append(local(link[0]))
    everything = {local(s) for t in (MFG.Machine, MFG.Buffer) for s in g.subjects(RDF.type, t)}
    return seq, sorted(everything - set(seq))


def flow_edges(g: Graph) -> list[tuple[str, str]]:
    return sorted((local(s), local(o)) for s, o in g.subject_objects(MFG.feeds) if isinstance(o, URIRef))


# ---------------------------------------------------------------- machines
MACHINE_PROPS = {
    "Name": RDFS.label, "Type": MFG.machineType, "Cycle time (s)": MFG.cycleTimeS,
    "Power (kW)": MFG.powerKW, "MTBF (h)": MFG.mtbfH, "MTTR (h)": MFG.mttrH,
}
NUMERIC = {"Cycle time (s)", "Power (kW)", "MTBF (h)", "MTTR (h)"}


def machine_table(g: Graph) -> pd.DataFrame:
    """One row per machine. A property with several values is shown as empty
    and listed under 'Conflicts' - the dashboard never picks one silently."""
    order = line_order(g)
    rows = []
    for m in g.subjects(RDF.type, MFG.Machine):
        row, conflicts = {"Machine": code_of(m)}, []
        for name, p in MACHINE_PROPS.items():
            vals = sorted({str(o) for o in g.objects(m, p)})
            if len(vals) == 1:
                row[name] = num(vals[0]) if name in NUMERIC else vals[0]
                if name in NUMERIC and row[name] is None:
                    conflicts.append(f"{name}: not a number ({vals[0]!r})")
            else:
                row[name] = None
                if len(vals) > 1:
                    conflicts.append(f"{name}: {' | '.join(vals)}")
        mtbf, mttr = row["MTBF (h)"], row["MTTR (h)"]
        row["Availability"] = round(mtbf / (mtbf + mttr), 4) if mtbf and mttr is not None else None
        row["Feeds"] = ", ".join(sorted(code_of(b) for b in g.objects(m, MFG.feeds)))
        row["Conflicts"] = "; ".join(conflicts)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_pos"] = df["Machine"].map({c: i for i, c in enumerate(order)})
    return df.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)


def bottleneck(df: pd.DataFrame):
    """Machine with the longest cycle time (only machines with one clear value)."""
    known = df.dropna(subset=["Cycle time (s)"])
    return None if known.empty else known.loc[known["Cycle time (s)"].idxmax(), "Machine"]


# ---------------------------------------------------------------- events
def failure_log(g: Graph) -> pd.DataFrame:
    def one(s, p):
        vals = sorted({code_of(o) if isinstance(o, URIRef) else str(o) for o in g.objects(s, p)})
        return " | ".join(vals) if vals else None

    rows = [{"Incident": code_of(f), "Machine": one(f, MFG.onMachine), "Failure mode": one(f, MFG.failureMode),
             "Downtime (h)": one(f, MFG.downtimeH), "Date": one(f, MFG.date), "Shift": one(f, MFG.duringShift),
             "Repaired by": ", ".join(sorted(code_of(a) for a in g.subjects(MFG.addresses, f))) or None}
            for f in g.subjects(RDF.type, MFG.Failure)]
    return pd.DataFrame(rows).sort_values(["Date", "Incident"], na_position="last").reset_index(drop=True) \
        if rows else pd.DataFrame()


def headline_metrics(g: Graph) -> dict:
    """The same handful of operator numbers, for comparing the three graphs."""
    mt = machine_table(g)
    dt = query(g, "downtime_by_machine")
    shifts = query(g, "failures_by_shift")
    actions = query(g, "actions_by_type")
    night = shifts.loc[shifts["shift"] == "NIGHT", "failures"].sum() if not shifts.empty else 0
    worst = dt.sort_values(["failures", "downtime_h"], ascending=False).iloc[0]["machine"] if not dt.empty else None
    return {
        "Machines": len(mt),
        "Bottleneck (longest cycle)": bottleneck(mt),
        "Total rated power (kW)": round(float(mt["Power (kW)"].sum()), 1) if not mt.empty else 0,
        "Failures": int(dt["failures"].sum()) if not dt.empty else 0,
        "Total downtime (h)": round(float(dt["downtime_h"].sum()), 2) if not dt.empty else 0,
        "Machine with most failures": worst,
        "Night-shift failures": int(night),
        "Preventive actions": int(actions.loc[actions["type"] == "Preventive", "actions"].sum()) if not actions.empty else 0,
        "Machines with conflicting data": int((mt["Conflicts"] != "").sum()) if not mt.empty else 0,
    }


# ---------------------------------------------------------------- data quality
def violations_df(g: Graph) -> pd.DataFrame:
    _conforms, v = validate_graph(g)
    df = pd.DataFrame(v)
    if df.empty:
        return pd.DataFrame(columns=["entity", "rule", "check", "property", "value", "message"])
    df["entity"] = df["focus"].str.split(":").str[-1]
    df["rule"] = df["shape"].str.split(":").str[-1]
    return df.rename(columns={"component": "check", "path": "property"})[
        ["entity", "rule", "check", "property", "value", "message"]]


def violations_per_entity(g: Graph) -> dict[str, int]:
    return dict(Counter(violations_df(g)["entity"]))


def silent_errors(g: Graph, gold: Graph) -> dict:
    """Split the graph's wrong facts into 'caught' (SHACL complains about this
    property of this entity) and 'silent' (no rule fires on it). Only possible
    because we have a ground truth.

    A wrong rdf:type fact counts as caught when its entity has any violation
    (an invented entity is flagged through its missing fields). A wrong
    'X feeds B' fact also counts as caught when buffer B has a fedBy violation."""
    s = score(g, gold)
    v = violations_df(g)
    flagged_props = {(e, p.split(":")[-1]) for e, p in zip(v["entity"], v["property"])}
    flagged_entities = set(v["entity"])
    fed_by = {e for e, p in zip(v["entity"], v["property"]) if p.startswith("^")}

    def is_caught(fact: str) -> bool:
        subj, pred, obj = fact.split(" ", 2)
        if pred == "type":
            return subj in flagged_entities
        if pred == "feeds" and obj.rsplit("/", 1)[-1] in fed_by:
            return True
        return (subj, pred) in flagged_props

    wrong = s["wrong_or_extra"]
    return {"score": s["overall"], "caught": [w for w in wrong if is_caught(w)],
            "silent": [w for w in wrong if not is_caught(w)], "missing": s["missing"]}


# ---------------------------------------------------------------- results files
def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def overview_numbers() -> dict:
    """Headline numbers for the Overview page, all read from results/ files."""
    p2 = load_json("phase2_scores.json")["overall"]
    rep = load_json("phase3_v2_repair.json")
    qa = load_json("phase4_scores.json")
    usage = load_json("llm_usage.json")["total"]
    schema = load_json("phase1_schema.json")
    return {
        "documents": len(list(DOCS.glob("*.txt"))),
        "triples_extracted": p2["extracted"],
        "triples_gold": p2["gold"],
        "f1_before": rep["rounds"][0]["f1"],
        "f1_after": rep["rounds"][-1]["f1"],
        "violations_before": rep["rounds"][0]["violations"],
        "violations_after": rep["rounds"][-1]["violations"],
        "repairs_attempted": rep["attempts"],
        "repairs_accepted": rep["accepted"],
        "shapes": schema["shapes"]["property_shapes"] + 1,  # + the SPARQL topology rule
        "constraints": schema["shapes"]["constraints_total"],
        "qa": {k: qa[k]["lenient"] for k in ["vector_rag", "graphrag_repaired", "graphrag_gold"]},
        "llm_calls": usage["calls"],
        "llm_hours": usage["hours"],
    }
