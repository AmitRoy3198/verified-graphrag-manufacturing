"""Live pipeline: give the system a new document and watch every step happen,
with the local open-source LLM. Each run is saved to its own folder in
dashboard_runs/; the research results in results/ are never changed."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from rdflib import RDF, URIRef  # noqa: E402

from dashboard import data, live, ui  # noqa: E402
from src.repair import code_of  # noqa: E402

ui.setup("Live pipeline", "▶️")
st.caption(ui.MODEL_LINE)

if os.environ.get("DASHBOARD_MODE") == "readonly":
    st.info("This dashboard runs in read-only mode (DASHBOARD_MODE=readonly), so live runs are switched off.")
    st.stop()

version = live.ollama_version()
if version is None:
    st.error("The local Ollama server is not running, so the LLM steps cannot start. Start **Ollama** from the "
             "Start menu (or run `ollama serve`), then reload this page.", icon="🛑")
    st.stop()
st.success(f"✓ Local Ollama {version} is running. Each LLM step takes about 30-90 s on this laptop "
           "(the model is reloaded for every call, for reproducible results).")

# ---------------------------------------------------------------- input
st.subheader("1 · The document")
samples = {p.stem: p for p in sorted(live.SAMPLES.glob("*.txt")) if p.name != "README.txt"}
how = st.radio("Input", ["Use a sample document", "Paste text", "Upload a .txt file"], horizontal=True)
if how == "Use a sample document":
    pick = st.selectbox("Sample (new documents, not part of the evaluation)", list(samples))
    text = samples[pick].read_text(encoding="utf-8")
    st.text_area("Document text", text, height=170, disabled=True)
elif how == "Paste text":
    text = st.text_area("Document text", height=170, placeholder="Paste a maintenance report, data sheet or shift note")
else:
    up = st.file_uploader("Text file", type=["txt"])
    text = up.read().decode("utf-8", errors="replace") if up else ""
    if text:
        st.text_area("Document text", text, height=170, disabled=True)

c1, c2 = st.columns(2)
do_repair = c1.toggle("Repair new rule violations with the LLM", value=True,
                      help="Targeted + gated repair (v2): only violated properties, multiple choice where possible, "
                           "kept only if the violation count drops.")
max_attempts = c2.slider("Maximum repair attempts", 1, 6, 3, disabled=not do_repair,
                         help="Each attempt is one LLM call (about 30-60 s).")

if st.button("▶ Run the pipeline on this document", type="primary", disabled=not text.strip()):
    run_dir = live.new_run_dir("document")
    base = data.load_graph(live.BASE)
    before = data.violations_df(base)
    result = {"run": run_dir.name}
    t0 = time.perf_counter()

    with st.status("Step 2 · Reading: the local LLM fills the extraction form ...", expanded=True) as s:
        ext, info = live.extract(text, run_dir)
        result["extraction"] = ext.model_dump(exclude_none=True, exclude_defaults=True)
        result["extract_seconds"] = info["seconds"]
        result["parse_error"] = info["parse_error"]
        s.update(label=f"Step 2 · Reading: done in {info['seconds']} s", state="complete" if not info["parse_error"]
                 else "error")

    with st.status("Step 3 · Organising: JSON → RDF triples ...") as s:
        doc_graph = live.to_graph(ext, run_dir)
        result["triples"] = sorted(f"{data.local(s)} {str(p).split('#')[-1]} "
                                   f"{data.local(o) if isinstance(o, URIRef) else o}"
                                   for s, p, o in doc_graph if p != RDF.type)
        s.update(label=f"Step 3 · Organising: {len(doc_graph)} triples", state="complete")

    with st.status("Step 4 · Checking: merging into the repaired graph and running SHACL ...") as s:
        merged = live.merge(base, doc_graph)
        after_merge = data.violations_df(merged)
        new = live.new_violations(before, after_merge)
        result["new_violations"] = new.to_dict("records")
        s.update(label=f"Step 4 · Checking: {len(new)} new rule violation(s)", state="complete")

    result["repairs"] = []
    if do_repair and len(new):
        with st.status("Step 5 · Fixing: targeted, gated repair with the local LLM ...", expanded=True) as s:
            result["repairs"] = live.repair(merged, doc_graph, text, run_dir, max_attempts=max_attempts)
            kept = sum(r["accepted"] for r in result["repairs"])
            s.update(label=f"Step 5 · Fixing: {len(result['repairs'])} attempt(s), {kept} kept by the gate",
                     state="complete")

    final = data.violations_df(merged)
    result["violations"] = {"base": len(before), "after_merge": len(after_merge), "final": len(final)}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    live.save(run_dir, merged, before, final, {k: v for k, v in result.items() if k != "extraction"})
    st.session_state.live_result = result
    ui.graph.clear()  # the new run appears in the graph picker on every page

# ---------------------------------------------------------------- result of the last run
r = st.session_state.get("live_result")
if r:
    st.divider()
    st.subheader(f"Result of run `{r['run']}`")
    v = r["violations"]
    c = st.columns(2) + st.columns(2)
    c[0].metric("Total time (s)", r["seconds"], border=True)
    c[1].metric("New facts", len(r["triples"]), border=True, help="Not counting rdf:type facts.")
    c[2].metric("Rule violations after merging", v["after_merge"], v["after_merge"] - v["base"],
                delta_color="inverse", border=True, help=f"The repaired graph had {v['base']} before.")
    c[3].metric("After repair", v["final"], v["final"] - v["after_merge"], delta_color="inverse", border=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**What the LLM extracted (JSON)**")
        if r["parse_error"]:
            st.error(f"The model's answer was not valid JSON: {r['parse_error'][:300]}")
        st.json(r["extraction"], expanded=True)
    with right:
        st.markdown("**As RDF facts in the graph**")
        st.dataframe(pd.DataFrame({"fact": r["triples"]}), hide_index=True, width="stretch", height=320)

    st.markdown("**New rule violations caused by this document**")
    if r["new_violations"]:
        st.dataframe(pd.DataFrame(r["new_violations"]), hide_index=True, width="stretch")
    else:
        st.success("✓ The new facts break no rule. (That does not prove they are true: SHACL checks form, not truth.)")

    if r["repairs"]:
        st.markdown("**Repair attempts**")
        for rec in r["repairs"]:
            ok = rec["accepted"]
            with st.container(border=True):
                st.markdown(f"**{rec['node']}** · violations {rec.get('violations_before', '?')} → "
                            f"{rec.get('violations_after', '?')}")
                answer = rec.get("answer") or {}
                kept, rolled = rec.get("kept_properties", []), rec.get("rolled_back_properties", [])
                for prop, p in rec["plan"].items():
                    gate = "✓ kept" if prop in kept else ("↩ rolled back" if prop in rolled else
                                                          ("✓ kept" if ok else "↩ rolled back"))
                    st.markdown(f"- {gate} · **{prop}**: model chose `{answer.get(prop)}` from "
                                f"{p['choices'] or 'a free value'} (was {p['current'] or 'empty'})")
                if answer.get("delete_entity"):
                    st.markdown(f"- model asked to delete the entity → {'✓ kept' if ok else '↩ rolled back'}")
    st.caption("Each property's change is gated on its own: it is kept only if it lowers the number of rule "
               "violations by itself. (The published results used one gate per entity; a live run showed that a "
               "good fix could then carry a bad one through.)")
    st.info(f"Saved to `dashboard_runs/{r['run']}/` (document, extraction, triples, merged graph, violations, "
            "repairs and every LLM call). Pick this run as the knowledge graph in the sidebar of any page to see "
            "its analytics.")

# ---------------------------------------------------------------- earlier runs
runs = live.live_runs()
if runs:
    with st.expander(f"Earlier live runs ({len(runs)})"):
        rows = []
        for p in runs:
            s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
            first = (p / "document.txt").read_text(encoding="utf-8").strip().splitlines()[0]
            rows.append({"run": p.name, "document": first, "seconds": s["seconds"],
                         "violations base → after merge → final":
                             f"{s['violations']['base']} → {s['violations']['after_merge']} → {s['violations']['final']}",
                         "repairs kept": sum(x["accepted"] for x in s["repairs"])})
        st.dataframe(rows, hide_index=True, width="stretch")
