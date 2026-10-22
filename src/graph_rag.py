"""GraphRAG: answer from a small part of the knowledge graph.

1. Entity linking (no LLM): find graph entities named in the question
   - codes like P2, B3, INC-204, WO-1104
   - labels like "Packer K1", "Night shift", "Line A"
2. Expansion: collect the entities up to 2 hops away from those seeds
   (following edges in both directions). Line and shift nodes are "hubs"
   (connected to almost everything), so we do not expand through them.
3. Questions that are about a whole class ("how many failures", "total power of
   all machines", "downstream of the bottleneck") also pull in every entity of
   that class, but only with their own facts (no expansion), to keep it small.
4. The facts of all collected entities become the LLM context, one line each:
   "machine_P2 powerKW 90.0". The LLM must answer from these lines only.
"""
import re

from rdflib import Graph, Literal, RDF, URIRef

from src.repair import code_of
from src.repair_v2 import local
from src.to_rdf import norm_id
from src.validate import EX, MFG

CODE = re.compile(r"[A-Za-z]+-?\d+")
CLASS_WORDS = {
    MFG.Failure: ["failure", "incident"],
    MFG.MaintenanceAction: ["maintenance", "work order"],
    MFG.Machine: ["machine", "station"],
    MFG.Buffer: ["buffer"],
    MFG.Shift: ["shift"],
}
AGGREGATION_CUES = ["how many", "total", "most", "highest", "longest", "lowest", "all "]
TOPOLOGY_CUES = ["bottleneck", "downstream", "upstream"]
HUBS = {MFG.ProductionLine, MFG.Shift}
MAX_LINES = 250  # keeps the prompt inside the context window


def entities(g: Graph) -> set:
    return {s for s in g.subjects(RDF.type, None) if str(s).startswith(str(EX))}


def link(g: Graph, question: str) -> tuple[set, set]:
    """Returns (specific seeds, class seeds)."""
    q = question.lower()
    ents = entities(g)
    seeds = set()
    codes = {norm_id(m) for m in CODE.findall(question)}
    for e in ents:
        code = code_of(e)
        if code in codes:  # P2, INC204, ...
            seeds.add(e)
        elif not any(ch.isdigit() for ch in code) and re.search(rf"\b{code.lower()}\b", q):
            seeds.add(e)  # names without digits: NIGHT, EARLY, ...
    for e in ents:
        for label in g.objects(e, URIRef("http://www.w3.org/2000/01/rdf-schema#label")):
            if len(str(label)) >= 4 and str(label).lower() in q:
                seeds.add(e)

    class_seeds = set()
    wants_class = not seeds or any(c in q for c in AGGREGATION_CUES + TOPOLOGY_CUES)
    if wants_class:
        for cls, words in CLASS_WORDS.items():
            if any(w in q for w in words):
                class_seeds |= set(g.subjects(RDF.type, cls))
    if any(c in q for c in TOPOLOGY_CUES):
        class_seeds |= set(g.subjects(RDF.type, MFG.Machine)) | set(g.subjects(RDF.type, MFG.Buffer))
    return seeds, class_seeds


def is_hub(g: Graph, node) -> bool:
    return any(t in HUBS for t in g.objects(node, RDF.type))


def neighbours(g: Graph, node) -> set:
    out = {o for o in g.objects(node, None) if isinstance(o, URIRef) and str(o).startswith(str(EX))}
    inc = {s for s in g.subjects(None, node) if str(s).startswith(str(EX))}
    return out | inc


def expand(g: Graph, seeds: set, hops: int = 2) -> set:
    nodes, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            if n in seeds or not is_hub(g, n):
                nxt |= neighbours(g, n)
        frontier = nxt - nodes
        nodes |= frontier
    return nodes


def fact_lines(g: Graph, nodes: set) -> list[str]:
    lines = []
    for s in nodes:
        for p, o in g.predicate_objects(s):
            if p == RDF.type:
                continue
            value = local(o) if isinstance(o, URIRef) else str(o) if isinstance(o, Literal) else str(o)
            lines.append(f"{local(s)} {str(p).split('#')[-1]} {value}")
    return sorted(set(lines))


def retrieve(g: Graph, question: str) -> tuple[str, dict]:
    """Returns (context text for the prompt, info about what was retrieved)."""
    seeds, class_seeds = link(g, question)
    nodes = expand(g, seeds) | class_seeds
    lines = fact_lines(g, nodes)
    info = {"seeds": sorted(local(s) for s in seeds), "class_seeds": len(class_seeds),
            "facts": len(lines), "truncated": max(0, len(lines) - MAX_LINES)}
    return "Facts:\n" + "\n".join(lines[:MAX_LINES]), info
