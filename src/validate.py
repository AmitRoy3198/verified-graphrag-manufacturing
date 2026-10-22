"""SHACL validation with pySHACL, returning violations as simple dicts.

The dicts are used by the tests (Phase 1) and sent to the LLM for repair (Phase 3).
"""
from pyshacl import validate as shacl_validate
from rdflib import BNode, Graph, Namespace, RDF

from src.config import SCHEMA

SH = Namespace("http://www.w3.org/ns/shacl#")
MFG = Namespace("http://example.org/mfg#")
SHP = Namespace("http://example.org/mfg/shapes#")
EX = Namespace("http://example.org/line/")


def load_shapes() -> Graph:
    return Graph().parse(SCHEMA / "shapes.ttl")


def load_ontology() -> Graph:
    return Graph().parse(SCHEMA / "ontology.ttl")


def _short(g: Graph, node) -> str:
    """Turn an IRI into prefix:name; render an inverse path as ^prop."""
    if node is None:
        return ""
    if isinstance(node, BNode):
        inv = g.value(node, SH.inversePath)
        return "^" + _short(g, inv) if inv is not None else "_:blank"
    try:
        return g.namespace_manager.normalizeUri(node)
    except Exception:
        return str(node)


def validate_graph(data: Graph, shapes: Graph | None = None) -> tuple[bool, list[dict]]:
    """Validate `data` against shapes.ttl. Returns (conforms, sorted list of violations)."""
    shapes = shapes if shapes is not None else load_shapes()
    conforms, report, _text = shacl_validate(
        data, shacl_graph=shapes, inference="none", abort_on_first=False
    )
    for prefix, ns in [("mfg", MFG), ("shp", SHP), ("ex", EX), ("sh", SH)]:
        report.bind(prefix, ns)

    violations = []
    for r in report.subjects(RDF.type, SH.ValidationResult):
        component = report.value(r, SH.sourceConstraintComponent)
        violations.append({
            "focus": _short(report, report.value(r, SH.focusNode)),
            "path": _short(report, report.value(r, SH.resultPath)),
            "value": str(report.value(r, SH.value) or ""),
            "shape": _short(report, report.value(r, SH.sourceShape)),
            # e.g. "MinCount" instead of sh:MinCountConstraintComponent
            "component": str(component).split("#")[-1].replace("ConstraintComponent", ""),
            "message": str(report.value(r, SH.resultMessage) or ""),
        })
    # Sort so the output order is the same on every run.
    violations.sort(key=lambda v: (v["focus"], v["shape"], v["component"], v["value"]))
    return conforms, violations
