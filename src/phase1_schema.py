"""Phase 1 summary: size of the ontology and shapes, plus one example violation.

Writes results/phase1_schema.json.   Run:  python -m src.phase1_schema
"""
import json
from collections import Counter

from rdflib import Graph, Literal, RDF
from rdflib.namespace import OWL, XSD

from src.config import RESULTS, ROOT
from src.validate import EX, MFG, SH, load_ontology, load_shapes, validate_graph

# SHACL parameters we count as "constraints"
CONSTRAINT_PARAMS = ["minCount", "maxCount", "datatype", "class", "in", "minExclusive",
                     "minInclusive", "maxInclusive", "lessThan", "sparql"]


def main() -> None:
    onto, shapes = load_ontology(), load_shapes()
    constraints = Counter()
    for param in CONSTRAINT_PARAMS:
        constraints[param] = len(list(shapes.subject_objects(SH[param])))

    # Example: what a violation looks like (this is what the LLM will see in Phase 3).
    g = Graph().parse(ROOT / "tests" / "fixtures" / "valid_line.ttl")
    g.set((EX.M1, MFG.powerKW, Literal("-5", datatype=XSD.decimal)))
    _, example = validate_graph(g, shapes)

    summary = {
        "ontology": {
            "classes": len(set(onto.subjects(RDF.type, OWL.Class))),
            "object_properties": len(set(onto.subjects(RDF.type, OWL.ObjectProperty))),
            "datatype_properties": len(set(onto.subjects(RDF.type, OWL.DatatypeProperty))),
        },
        "shapes": {
            "node_shapes": len(set(shapes.subjects(RDF.type, SH.NodeShape))),
            "property_shapes": len(set(shapes.subjects(RDF.type, SH.PropertyShape))),
            "constraints_by_type": dict(constraints),
            "constraints_total": sum(constraints.values()),
        },
        "valid_fixture_conforms": validate_graph(Graph().parse(ROOT / "tests" / "fixtures" / "valid_line.ttl"), shapes)[0],
        "example_violation_powerKW_minus_5": example,
    }
    (RESULTS / "phase1_schema.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
