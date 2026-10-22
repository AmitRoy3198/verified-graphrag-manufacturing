"""Phase 1 tests: the shapes accept a valid graph, and every shape can FAIL.

How it works: load the valid fixture, break exactly one thing, validate, and
check that the *expected* shape reports the *expected* kind of violation.
A last test checks that the CASES table covers every shape in shapes.ttl,
so a new shape without a failing test makes the suite fail.
"""
from pathlib import Path

import pytest
from rdflib import Graph, Literal, RDF, RDFS
from rdflib.namespace import XSD

from src.validate import EX, MFG, SH, load_ontology, load_shapes, validate_graph

FIXTURE = Path(__file__).parent / "fixtures" / "valid_line.ttl"
SHAPES = load_shapes()


def fresh() -> Graph:
    return Graph().parse(FIXTURE)


def set_value(g, s, p, value):
    """Replace all values of (s, p) with one new value."""
    g.remove((s, p, None))
    g.add((s, p, value))


def dec(x):
    return Literal(str(x), datatype=XSD.decimal)


# (test id, function that breaks the graph, expected shape, expected component)
CASES = [
    # --- Machine ---
    ("machine_missing_label",      lambda g: g.remove((EX.M1, RDFS.label, None)),                 "shp:Machine-label", "MinCount"),
    ("machine_type_not_allowed",   lambda g: set_value(g, EX.M1, MFG.machineType, Literal("Robot")), "shp:Machine-machineType", "In"),
    ("cycle_time_zero",            lambda g: set_value(g, EX.M1, MFG.cycleTimeS, dec(0)),          "shp:Machine-cycleTime", "MinExclusive"),
    ("cycle_time_two_values",      lambda g: g.add((EX.M1, MFG.cycleTimeS, dec(45))),              "shp:Machine-cycleTime", "MaxCount"),
    ("power_negative",             lambda g: set_value(g, EX.M1, MFG.powerKW, dec(-5)),            "shp:Machine-power", "MinExclusive"),
    ("power_missing",              lambda g: g.remove((EX.M1, MFG.powerKW, None)),                 "shp:Machine-power", "MinCount"),
    ("power_as_text",              lambda g: set_value(g, EX.M1, MFG.powerKW, Literal("55 kW")),   "shp:Machine-power", "Datatype"),
    ("mtbf_missing",               lambda g: g.remove((EX.M1, MFG.mtbfH, None)),                   "shp:Machine-mtbf", "MinCount"),
    ("mttr_not_less_than_mtbf",    lambda g: set_value(g, EX.M1, MFG.mttrH, dec(200)),             "shp:Machine-mttr", "LessThan"),
    ("part_of_line_wrong_class",   lambda g: set_value(g, EX.M1, MFG.partOfLine, EX.B1),           "shp:Machine-partOfLine", "Class"),
    ("machine_feeds_machine",      lambda g: set_value(g, EX.M1, MFG.feeds, EX.M2),                "shp:Machine-feeds", "Class"),
    ("topology_machine_feeds_nothing", lambda g: g.remove((EX.M1, MFG.feeds, None)),              "shp:MachineTopologyShape", "SPARQL"),
    # --- Buffer ---
    ("buffer_missing_label",       lambda g: g.remove((EX.B1, RDFS.label, None)),                 "shp:Buffer-label", "MinCount"),
    ("buffer_capacity_zero",       lambda g: set_value(g, EX.B1, MFG.capacity, Literal(0)),        "shp:Buffer-capacity", "MinExclusive"),
    ("buffer_capacity_decimal",    lambda g: set_value(g, EX.B1, MFG.capacity, dec(20.5)),         "shp:Buffer-capacity", "Datatype"),
    ("buffer_feeds_nothing",       lambda g: g.remove((EX.B1, MFG.feeds, None)),                   "shp:Buffer-feeds", "MinCount"),
    ("buffer_fed_by_two_machines", lambda g: g.add((EX.M2, MFG.feeds, EX.B1)),                     "shp:Buffer-fedBy", "MaxCount"),
    # --- Failure ---
    ("failure_on_non_machine",     lambda g: set_value(g, EX.F1, MFG.onMachine, EX.S1),            "shp:Failure-onMachine", "Class"),
    ("failure_missing_mode",       lambda g: g.remove((EX.F1, MFG.failureMode, None)),             "shp:Failure-mode", "MinCount"),
    ("failure_downtime_negative",  lambda g: set_value(g, EX.F1, MFG.downtimeH, dec(-1)),          "shp:Failure-downtime", "MinExclusive"),
    ("failure_date_as_text",       lambda g: set_value(g, EX.F1, MFG.date, Literal("2 March")),    "shp:Failure-date", "Datatype"),
    ("failure_shift_wrong_class",  lambda g: set_value(g, EX.F1, MFG.duringShift, EX.M1),          "shp:Failure-shift", "Class"),
    # --- MaintenanceAction ---
    ("action_missing_machine",     lambda g: g.remove((EX.A1, MFG.onMachine, None)),               "shp:Maintenance-onMachine", "MinCount"),
    ("action_type_unknown",        lambda g: set_value(g, EX.A1, MFG.actionType, Literal("Repair")), "shp:Maintenance-type", "In"),
    ("action_missing_date",        lambda g: g.remove((EX.A1, MFG.date, None)),                    "shp:Maintenance-date", "MinCount"),
    ("action_duration_zero",       lambda g: set_value(g, EX.A1, MFG.durationH, dec(0)),           "shp:Maintenance-duration", "MinExclusive"),
    ("action_addresses_machine",   lambda g: set_value(g, EX.A1, MFG.addresses, EX.M1),            "shp:Maintenance-addresses", "Class"),
    # --- Shift ---
    ("shift_missing_label",        lambda g: g.remove((EX.S1, RDFS.label, None)),                 "shp:Shift-label", "MinCount"),
    ("shift_start_hour_25",        lambda g: set_value(g, EX.S1, MFG.startHour, Literal(25)),      "shp:Shift-startHour", "MaxInclusive"),
    # --- ProductionLine ---
    ("line_two_labels",            lambda g: g.add((EX.Line1, RDFS.label, Literal("Line A"))),    "shp:Line-label", "MaxCount"),
    ("line_missing_final_machine", lambda g: g.remove((EX.Line1, MFG.finalMachine, None)),         "shp:Line-finalMachine", "MinCount"),
]


def test_valid_fixture_conforms():
    conforms, violations = validate_graph(fresh(), SHAPES)
    assert conforms, violations


@pytest.mark.parametrize("name,break_it,shape,component", CASES, ids=[c[0] for c in CASES])
def test_shape_can_fail(name, break_it, shape, component):
    g = fresh()
    break_it(g)
    conforms, violations = validate_graph(g, SHAPES)
    assert not conforms, f"{name}: graph was broken but SHACL still conforms"
    fired = {(v["shape"], v["component"]) for v in violations}
    assert (shape, component) in fired, f"{name}: expected {shape}/{component}, got {fired}"


def test_every_shape_has_a_failing_case():
    """A shape that is never shown to fail could be silently broken."""
    nm = SHAPES.namespace_manager
    all_shapes = {nm.normalizeUri(p) for p in SHAPES.objects(None, SH.property)}
    all_shapes |= {nm.normalizeUri(s) for s in SHAPES.subjects(SH.sparql, None)}
    tested = {c[2] for c in CASES}
    assert all_shapes - tested == set(), f"shapes without a failing test: {all_shapes - tested}"


def test_shapes_only_use_ontology_terms():
    """Every class/property in the shapes must be declared in ontology.ttl."""
    onto = load_ontology()
    declared = set(onto.subjects(RDF.type, None))
    used = set(SHAPES.objects(None, SH.targetClass)) | set(SHAPES.objects(None, SH["class"]))
    used |= {p for p in SHAPES.objects(None, SH.path) if str(p).startswith(str(MFG))}
    missing = {u for u in used if str(u).startswith(str(MFG)) and u not in declared}
    assert not missing, f"used in shapes but not declared in ontology: {missing}"
