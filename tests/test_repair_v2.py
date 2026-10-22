"""Phase 3 v2 tests: targeting, multiple choice and the gate, with a fake LLM."""
from pathlib import Path

import pytest
from pydantic import ValidationError
from rdflib import Graph, Literal, RDFS
from rdflib.namespace import XSD

from src.repair_v2 import fix_model, gated_repair, plan_fix, prop_name
from src.validate import EX, MFG, validate_graph

FIXTURE = Path(__file__).parent / "fixtures" / "valid_line.ttl"


def broken_two_labels():
    g = Graph().parse(FIXTURE)
    g.add((EX.M1, RDFS.label, Literal("Press")))  # second label -> MaxCount violation
    return g


def violations_of(g, node="ex:M1"):
    return [v for v in validate_graph(g)[1] if v["focus"] == node]


def test_prop_name_mapping():
    assert prop_name({"path": "mfg:powerKW"}) == "powerKW"
    assert prop_name({"path": "^mfg:feeds"}) == "fedBy"
    assert prop_name({"path": ""}) == "feeds"  # SPARQL topology rule


def test_plan_targets_only_violated_property_and_offers_choices():
    g = broken_two_labels()
    plan = plan_fix(g, EX.M1, MFG.Machine, violations_of(g))
    assert list(plan) == ["label"]  # power, cycle time etc. are locked
    assert plan["label"]["choices"] == ["Press", "Press 1"]


def test_schema_is_multiple_choice():
    g = broken_two_labels()
    Model = fix_model(plan_fix(g, EX.M1, MFG.Machine, violations_of(g)))
    assert set(Model.model_fields) == {"label", "delete_entity"}
    Model(label="Press 1", delete_entity=False)  # an option: fine
    with pytest.raises(ValidationError):
        Model(label="Something new", delete_entity=False)  # not an option: rejected


def test_good_fix_is_accepted():
    g = broken_two_labels()
    rec = gated_repair(g, EX.M1, violations_of(g), lambda inputs, model: {"label": "Press 1", "delete_entity": False})
    assert rec["accepted"] and validate_graph(g)[0]
    assert (EX.M1, MFG.powerKW, Literal("55.0", datatype=XSD.decimal)) in g  # locked fact untouched


def test_fix_that_does_not_help_is_rolled_back():
    g = broken_two_labels()
    snapshot = set(g)
    # answering null removes both labels -> MinCount violation instead of MaxCount: no improvement
    rec = gated_repair(g, EX.M1, violations_of(g), lambda inputs, model: {"label": None, "delete_entity": False})
    assert not rec["accepted"]
    assert set(g) == snapshot  # graph unchanged


def test_harmful_deletion_is_rolled_back():
    g = broken_two_labels()
    snapshot = set(g)
    # deleting M1 removes its violation but breaks the buffer, the failure and the action
    rec = gated_repair(g, EX.M1, violations_of(g), lambda inputs, model: {"label": None, "delete_entity": True})
    assert not rec["accepted"] and set(g) == snapshot


def test_no_answer_leaves_graph_unchanged():
    g = broken_two_labels()
    snapshot = set(g)
    rec = gated_repair(g, EX.M1, violations_of(g), lambda inputs, model: None)
    assert not rec["accepted"] and set(g) == snapshot
