"""Phase 4 tests: answer scoring and GraphRAG retrieval (no LLM needed)."""
from rdflib import Graph

from src.config import DATA
from src.graph_rag import retrieve
from src.qa import exact_match, lenient_match, load_questions

Q = {q["id"]: q for q in load_questions()}
GOLD = Graph().parse(DATA / "ground_truth.ttl")


def test_number_scoring():
    q = Q["Q01"]  # 90 kW
    assert exact_match("90", q) and exact_match("90.0 kW", q)
    assert not exact_match("Press P2 has 90 kW", q)  # not a bare number
    assert lenient_match("Press P2 has 90 kW", q)  # the 2 in "P2" is not taken as the number
    assert not lenient_match("75", q)
    assert lenient_match("0.76", Q["Q03"])  # within 2 % of 0.75
    assert not lenient_match("45", Q["Q03"])  # minutes not converted


def test_entity_scoring():
    q = Q["Q09"]  # INC-204
    assert exact_match("INC-204", q) and exact_match("inc204", q)
    assert lenient_match("failure_INC204", q)
    assert not lenient_match("INC-201", q)
    assert exact_match("Packer K1", Q["Q10"]) and lenient_match("machine_K1", Q["Q10"])
    assert not lenient_match("K10", Q["Q10"])


def test_text_scoring():
    q = Q["Q07"]
    assert exact_match("Die misalignment", q)
    assert lenient_match("die misalignment on the press", q) and not exact_match("die misalignment on the press", q)


def test_graph_context_contains_the_needed_facts():
    """On the ground-truth graph, retrieval must reach the facts each multi-hop answer needs."""
    needed = {
        "Q12": ["machine_P2 feeds buffer_B2", "buffer_B2 feeds machine_W1"],
        "Q16": ["action_WO1108 addresses failure_INC208", "failure_INC208 duringShift shift_NIGHT"],
        "Q17": ["failure_INC204 onMachine machine_W1", "machine_W1 mttrH 1.5"],
        "Q18": ["buffer_B5 feeds machine_K1", "machine_I1 feeds buffer_B5", "machine_I1 cycleTimeS 30.0"],
    }
    for qid, facts in needed.items():
        context, _ = retrieve(GOLD, Q[qid]["question"])
        for f in facts:
            assert f in context, f"{qid}: missing '{f}'"
