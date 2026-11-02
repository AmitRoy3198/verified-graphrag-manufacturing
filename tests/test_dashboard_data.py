"""Dashboard data layer: on the ground truth, the operator numbers must equal
the SPARQL-verified answers from data/questions.json (no LLM involved)."""
from dashboard.data import (SOURCES, failure_log, headline_metrics, line_order, load_graph,
                            machine_table, overview_numbers, silent_errors, violations_df)
import pandas as pd

from src.config import DATA

GOLD = load_graph(DATA / "ground_truth.ttl")
EXTRACTED = load_graph(SOURCES["Extracted graph (before repair)"])


def test_line_order_follows_material_flow():
    assert line_order(GOLD) == ["P1", "P2", "W1", "A1", "I1", "K1"]


def test_headline_metrics_match_verified_answers():
    m = headline_metrics(GOLD)
    assert m["Bottleneck (longest cycle)"] == "A1"          # Q19
    assert m["Total rated power (kW)"] == 246               # Q24
    assert m["Total downtime (h)"] == 20                    # Q25
    assert m["Machine with most failures"] == "A1"          # Q22
    assert m["Night-shift failures"] == 4                   # Q20
    assert m["Preventive actions"] == 2                     # Q23
    assert m["Machines with conflicting data"] == 0


def test_machine_table_availability_and_conflicts():
    gold = machine_table(GOLD).set_index("Machine")
    assert round(gold.loc["P2", "Availability"], 4) == round(140 / 144, 4)
    ext = machine_table(EXTRACTED).set_index("Machine")
    assert "Name: Press | Press P2" in ext.loc["P2", "Conflicts"]   # never picks one silently
    assert pd.isna(ext.loc["P2", "Name"])


def test_failure_log_and_quality():
    assert len(failure_log(GOLD)) == 8
    assert violations_df(GOLD).empty
    assert len(violations_df(EXTRACTED)) == 54
    q = silent_errors(EXTRACTED, GOLD)
    assert "machine_I1 mttrH 45.0" in q["silent"]       # the unconverted 45 min: SHACL cannot see it
    assert len(q["caught"]) + len(q["silent"]) == q["score"]["extracted"] - q["score"]["correct"]


def test_overview_numbers_come_from_results():
    o = overview_numbers()
    assert (o["documents"], o["violations_before"], o["violations_after"]) == (21, 54, 21)
    assert (o["f1_before"], o["f1_after"]) == (0.742, 0.767)


def test_d2_process_data_matches_results():
    from dashboard.data import llm_log, per_predicate_f1, qa_answers, repair_attempts_v2, repair_rounds
    rounds = repair_rounds()
    assert list(rounds.groupby("design")["violations"].last().sort_values()) == [21, 71]
    att = repair_attempts_v2()
    assert len(att) == 49 and (att["status"] == "✓ kept").sum() == 18
    assert per_predicate_f1().set_index("property").loc["feeds", "after v2 repair"] == 0.952
    qa = qa_answers()
    assert len(qa) == 100  # 25 questions x 4 conditions
    lenient = qa.groupby("condition")["lenient"].mean().round(2).to_dict()
    assert lenient == {"Vector RAG": 0.52, "GraphRAG · unrepaired": 0.36,
                       "GraphRAG · repaired": 0.4, "GraphRAG · perfect graph": 0.64}
    log = llm_log()
    assert len(log) == 354 and round(log["seconds"].max()) == 8439
