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
