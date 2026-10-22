"""Phase 2 tests for the deterministic parts (no LLM needed)."""
from rdflib import Graph

from src.config import DATA
from src.extract_schema import Buffer, Extraction, Failure, Machine
from src.score import score
from src.to_rdf import extraction_to_graph, norm_id
from src.validate import validate_graph

GOLD = Graph().parse(DATA / "ground_truth.ttl")


def test_ground_truth_conforms_to_shapes():
    conforms, violations = validate_graph(GOLD)
    assert conforms, violations


def test_norm_id():
    assert norm_id("Press P2") == "P2"
    assert norm_id("INC-205") == "INC205"
    assert norm_id("WO-1101") == "WO1101"
    assert norm_id("Night shift") == "NIGHT"
    assert norm_id("Line A") == "LINEA"
    assert norm_id("Assembly cell A1") == "A1"


def test_gold_against_itself_is_perfect():
    assert score(GOLD, GOLD)["overall"]["f1"] == 1.0


def test_to_rdf_and_scoring_on_a_small_example():
    ex = Extraction(
        machines=[Machine(id="P1", name="Press P1", power_kw=75, feeds_buffer="B1")],
        buffers=[Buffer(id="Buffer B1", capacity=40.0, feeds_machine="Press P2")],
        failures=[Failure(id="INC-201", machine="P2", date="3 March 2026")],  # bad date format
    )
    g = extraction_to_graph(ex)
    s = score(g, GOLD)["overall"]
    # correct: P1 type, label, power, feeds B1; B1 type, capacity, feeds P2; INC201 type, onMachine = 9
    assert s["correct"] == 9
    assert s["extracted"] == 10  # the badly formatted date is the one wrong triple
    # ...and SHACL sees the bad date as a datatype violation
    _, violations = validate_graph(g)
    assert any(v["shape"] == "shp:Failure-date" and v["component"] == "Datatype" for v in violations)
