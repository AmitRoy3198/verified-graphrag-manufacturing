"""Phase D3 live logic, tested without the LLM (a fake model answers)."""
import src.llm as llm
from dashboard import data, live
from src.extract_schema import Extraction, Machine


def test_new_run_dir_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "RUNS", tmp_path)
    d = live.new_run_dir("document")
    assert d.exists() and d.name.endswith("_document")


def test_log_to_restores_the_project_log(tmp_path):
    original = llm.LOG_FILE
    with live.log_to(tmp_path / "x.jsonl"):
        assert llm.LOG_FILE == tmp_path / "x.jsonl"
    assert llm.LOG_FILE == original


def test_conflicting_new_value_is_caught_and_repaired(tmp_path):
    """The W1 retrofit sample: a second power value (52 kW) -> SHACL MaxCount -> multiple choice."""
    base = data.load_graph(live.BASE)
    before = data.violations_df(base)
    doc = live.to_graph(Extraction(machines=[Machine(id="W1", power_kw=52)]), tmp_path)
    merged = live.merge(base, doc)
    new = live.new_violations(before, data.violations_df(merged))
    assert list(zip(new["entity"], new["rule"], new["check"])) == [("machine_W1", "Machine-power", "MaxCount")]

    asked = {}

    def fake_llm(inputs, model):
        asked["documents"] = inputs["documents"]
        asked["fields"] = set(model.model_fields)
        return {"powerKW": "52.0", "delete_entity": False}

    recs = live.repair(merged, doc, "New rated power of Washer W1: 52 kW.", tmp_path, ask=fake_llm)
    assert len(recs) == 1 and recs[0]["accepted"]
    assert asked["fields"] == {"powerKW", "delete_entity"}            # only the violated property
    assert asked["documents"].startswith("[new document]\nNew rated power")  # the model sees the new text
    assert len(data.violations_df(merged)) == len(before)              # back to the base count
    assert (tmp_path / "repairs.json").exists()


def test_finished_live_run_becomes_a_graph_source(tmp_path, monkeypatch):
    run = tmp_path / "dashboard_runs" / "20260101T000000Z_document"
    run.mkdir(parents=True)
    (run / "graph_after.ttl").write_text("", encoding="utf-8")
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    sources = data.all_sources()
    assert any(k.startswith("Live run 20260101T000000Z") for k in sources)


def test_live_repair_gates_each_property(tmp_path):
    """The live W1 case: line fix is good, 'no power value' is bad -> only the line fix is kept."""
    from src.extract_schema import Line
    base = data.load_graph(live.BASE)
    doc = live.to_graph(Extraction(machines=[Machine(id="W1", power_kw=52, line="A")],
                                   lines=[Line(id="A", name="Line A")]), tmp_path)
    merged = live.merge(base, doc)
    recs = live.repair(merged, doc, "Washer W1 on Line A: new rated power 52 kW.", tmp_path,
                       ask=lambda i, m: {"partOfLine": "LINEA", "powerKW": None, "delete_entity": False}
                       if "powerKW" in m.model_fields else None)
    w1 = next(r for r in recs if r["node"] == "machine_W1")
    assert w1["kept_properties"] == ["partOfLine"] and w1["rolled_back_properties"] == ["powerKW"]
    assert len(list(merged.objects(data.EX.machine_W1, data.MFG.powerKW))) == 2   # both values kept, still flagged


def w1_conflict_run(tmp_path):
    """A run folder for the W1 retrofit note: merged graph with 45 kW and 52 kW."""
    base = data.load_graph(live.BASE)
    doc = live.to_graph(Extraction(machines=[Machine(id="W1", power_kw=52)]), tmp_path)
    merged = live.merge(base, doc)
    merged.serialize(tmp_path / "graph_after.ttl", format="turtle")
    new = live.new_violations(data.violations_df(base), data.violations_df(merged))
    return base, doc, merged, new


def test_facts_are_sorted_into_new_known_and_conflict(tmp_path):
    base, doc, _, new = w1_conflict_run(tmp_path)
    rows = live.fact_rows(base, doc, new)
    power = next(r for r in rows if r["property"] == "rated power (kW)")
    assert power["status"] == "conflict" and power["value"] == "52.0" and power["our records say"] == "45.0"
    assert rows[0]["status"] == "conflict"                         # conflicts are listed first


def test_operator_question_offers_both_values_with_their_source(tmp_path):
    _, doc, merged, _ = w1_conflict_run(tmp_path)
    q = live.open_questions(merged, doc)
    assert [(x["entity"], x["property"]) for x in q] == [("machine_W1", "powerKW")]
    assert q[0]["choices"] == [{"value": "45.0", "source": "our records"},
                               {"value": "52.0", "source": "this document"}]


def test_operator_decision_is_applied_and_logged_in_the_run_folder(tmp_path):
    _, doc, merged, _ = w1_conflict_run(tmp_path)
    before = len(data.violations_df(merged))
    rec = live.decide(tmp_path, "machine_W1", "powerKW", "52.0")
    g = data.load_graph(tmp_path / "graph_after.ttl")
    assert [float(x) for x in g.objects(data.EX.machine_W1, data.MFG.powerKW)] == [52.0]
    assert rec["violations_after"] == before - 1 and live.open_questions(g, doc) == []
    assert live.decisions(tmp_path)[0]["chosen"] == "52.0"
    assert (tmp_path / "graph_before_decisions.ttl").exists()     # the AI's version is kept


def test_agent_calls_rebuild_the_exact_messages(tmp_path):
    (tmp_path / "document.txt").write_text("Washer W1: 52 kW.", encoding="utf-8")
    with live.log_to(tmp_path / "llm_log.jsonl"):
        llm.log_llm_call("phase2", "extract", {"doc": "document.txt"}, '{"machines": []}', 1.0)
    call = live.agent_calls(tmp_path)[0]
    assert call["agent"] == "Reader agent" and "Washer W1: 52 kW." in call["user"]
    assert call["output"] == '{"machines": []}'
