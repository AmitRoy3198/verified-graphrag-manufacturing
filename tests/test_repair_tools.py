"""Phase 3 tests for the deterministic repair helpers (no LLM needed)."""
from pathlib import Path

from rdflib import Graph, Literal
from rdflib.namespace import XSD

from src.repair import Fact, Repair, apply_repair, current_facts, to_value
from src.validate import EX, MFG, validate_graph

FIXTURE = Path(__file__).parent / "fixtures" / "valid_line.ttl"


def test_to_value_conversions():
    assert to_value("decimal", "0.75") == Literal("0.75", datatype=XSD.decimal)
    assert to_value("whole", "40") == Literal(40)
    assert to_value("date", "2026-03-03").datatype == XSD.date
    assert to_value("date", "3 March 2026").datatype is None  # stays text -> SHACL flags it
    assert to_value("decimal", "about 5") == Literal("about 5")  # not a number -> SHACL flags it
    assert to_value("machine", "Press P2") == EX.machine_P2


def test_current_facts_shows_codes_and_incoming_feeds_for_buffers():
    g = Graph().parse(FIXTURE)
    facts = current_facts(g, EX.B1, MFG.Buffer)
    assert ("feeds", "M2") in facts and ("fedBy", "M1") in facts


def test_repair_fixes_a_violation():
    g = Graph().parse(FIXTURE)
    g.add((EX.M1, MFG.powerKW, Literal("-5", datatype=XSD.decimal)))  # two values, one negative
    assert not validate_graph(g)[0]
    facts = [Fact(property=k, value=v) for k, v in current_facts(g, EX.M1, MFG.Machine)
             if not (k == "powerKW" and v == "-5")]
    # simulate an LLM answer that keeps every fact except the bad power value
    apply_repair(g, EX.M1, MFG.Machine, Repair(delete_entity=False, facts=facts))
    assert (EX.M1, MFG.powerKW, Literal("55.0", datatype=XSD.decimal)) in g
    assert (EX.M1, MFG.powerKW, Literal("-5", datatype=XSD.decimal)) not in g


def test_delete_entity_removes_all_references():
    g = Graph().parse(FIXTURE)
    notes = apply_repair(g, EX.F1, MFG.Failure, Repair(delete_entity=True, facts=[]))
    assert notes == ["entity deleted"]
    assert not list(g.triples((EX.F1, None, None)))
    assert not list(g.triples((None, None, EX.F1)))  # the action no longer addresses it


def test_unknown_property_is_ignored_and_reported():
    g = Graph().parse(FIXTURE)
    notes = apply_repair(g, EX.S1, MFG.Shift, Repair(delete_entity=False, facts=[
        Fact(property="label", value="Early shift"), Fact(property="colour", value="blue")]))
    assert notes == ["ignored unknown property 'colour'"]
