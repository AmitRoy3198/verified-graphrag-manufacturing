"""Live pipeline: give the system a new document and follow it step by step.

Five steps, and the page always says who does each one: two AI agents (the local
open-source LLM), two pieces of plain code/rules, and the operator. Every run is
saved to its own folder in dashboard_runs/ and can be opened again here; the
research results in results/ are never changed."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, live, ui  # noqa: E402

ui.setup("Live pipeline", "▶️")
st.markdown("Give the factory a new document and follow it through five steps. Two steps are **AI agents** "
            "(the local Qwen model), two are **plain rules and code** (no AI), and the last one is **you**.")
st.caption(ui.MODEL_LINE)
readonly = os.environ.get("DASHBOARD_MODE") == "readonly"

STEPS = [("📄", "Read", "Reader agent · AI"), ("🧩", "Understand", "Organiser · code"),
         ("🔎", "Check", "Rule checker · SHACL, no AI"), ("🛠️", "Fix", "Repair agent · AI + gate"),
         ("👤", "You decide", "Operator")]
STATE = {"wait": "⚪ waiting", "work": "⏳ working …", "done": "✅ done", "warn": "⚠️ needs attention",
         "you": "👤 your turn", "skip": "➖ skipped"}
STATUS = {"new": "🆕 new", "already known": "✔️ already known", "conflict": "⚠️ conflict"}


def draw_steps(slots, states, notes):
    for slot, (icon, name, who), state, note in zip(slots, STEPS, states, notes):
        with slot.container(border=True):
            st.markdown(f"**{icon} {name}**  \n:gray[{who}]  \n{STATE[state]}  \n{note}")


def look_inside(call):
    """Exactly what one agent received and answered, straight from the run's LLM log."""
    with st.expander(f"🔍 Look inside the {call['agent']}: what it received and what it answered"):
        st.caption(f"{call['model']} · temperature {call['temperature']} · seed {call['seed']} · "
                   f"{call['seconds']} s. The answer format is also enforced by a JSON schema (constrained "
                   "decoding): the agent can only answer with the fields it is given.")
        a, b, c = st.tabs(["Its instructions", "Its message", "Its answer"])
        a.code(call["system"], language=None, wrap_lines=True)
        b.code(call["user"], language=None, wrap_lines=True)
        c.code(call["output"], language="json", wrap_lines=True)


# ---------------------------------------------------------------- 1 · input
version = None if readonly else live.ollama_version()
with st.container(border=True):
    st.markdown("**Choose a document**")
    samples = {p.stem: p for p in sorted(live.SAMPLES.glob("*.txt")) if p.name != "README.txt"}
    how = st.radio("Input", ["Sample document", "Paste text", "Upload a .txt file"], horizontal=True,
                   label_visibility="collapsed")
    if how == "Sample document":
        pick = st.selectbox("Sample (new documents, not part of the evaluation)", list(samples))
        text = samples[pick].read_text(encoding="utf-8")
        st.text_area("Document text", text, height=130, disabled=True)
    elif how == "Paste text":
        text = st.text_area("Document text", height=130,
                            placeholder="Paste a maintenance report, data sheet or shift note")
    else:
        up = st.file_uploader("Text file", type=["txt"])
        text = up.read().decode("utf-8", errors="replace") if up else ""
    with st.expander("Settings"):
        do_repair = st.toggle("Let the Repair agent fix new rule problems", value=True)
        max_attempts = st.slider("Maximum repair attempts (one AI call each, about 30-60 s)", 1, 6, 3,
                                 disabled=not do_repair)
    if readonly:
        st.info("Read-only mode (DASHBOARD_MODE=readonly): new runs and decisions are switched off.")
    elif version is None:
        st.warning("The local Ollama server is not running, so the AI agents cannot start. Start **Ollama** from "
                   "the Start menu (or run `ollama serve`) and reload this page. Earlier runs can still be opened.",
                   icon="🛑")
    else:
        st.caption(f"✓ Local Ollama {version} is running. Each AI step takes about 30-90 s on this laptop.")
    clicked = st.button("▶ Run the five steps on this document", type="primary",
                        disabled=readonly or version is None or not text.strip())

# ---------------------------------------------------------------- 2 · the five steps (live while running)
st.subheader("The five steps")
picker = st.container()  # the run selector goes here, above the steps it describes
slots = [c.empty() for c in st.columns(5)]

if clicked:
    states, notes = ["work"] + ["wait"] * 4, [""] * 5
    draw_steps(slots, states, notes)
    run_dir = live.new_run_dir("document")
    base = data.load_graph(live.BASE)
    before = data.violations_df(base)
    result = {"run": run_dir.name}
    t0 = time.perf_counter()

    ext, info = live.extract(text, run_dir)
    result.update(extract_seconds=info["seconds"], parse_error=info["parse_error"])
    states[0:2], notes[0] = ["warn" if info["parse_error"] else "done", "work"], f"{info['seconds']} s"
    draw_steps(slots, states, notes)

    doc_graph = live.to_graph(ext, run_dir)
    result["triples"] = [f"{data.local(s)} {live.prop_of(p)} {data.local(o)}" for s, p, o in doc_graph
                         if p != live.RDF.type]
    states[1:3], notes[1] = ["done", "work"], f"{len(result['triples'])} facts"
    draw_steps(slots, states, notes)

    merged = live.merge(base, doc_graph)
    after_merge = data.violations_df(merged)
    new = live.new_violations(before, after_merge)
    result["new_violations"] = new.to_dict("records")
    states[2], notes[2] = ("warn" if len(new) else "done"), f"{len(new)} new problem(s)"
    result["repairs"] = []
    if do_repair and len(new):
        states[3] = "work"
        draw_steps(slots, states, notes)
        result["repairs"] = live.repair(merged, doc_graph, text, run_dir, max_attempts=max_attempts)

    final = data.violations_df(merged)
    result["violations"] = {"base": len(before), "after_merge": len(after_merge), "final": len(final)}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    live.save(run_dir, merged, before, final, result)
    st.session_state.live_run = run_dir.name
    ui.graph.clear()  # the new run appears in the graph picker on every page
    st.rerun()

# ---------------------------------------------------------------- 3 · open a run
runs = live.live_runs()
if not runs:
    draw_steps(slots, ["wait"] * 5, [""] * 5)
    st.info("No run yet. Choose a document above and press ▶ to watch the five steps.")
    st.stop()

names = [p.name for p in runs]
first_line = {p.name: (p / "document.txt").read_text(encoding="utf-8").strip().splitlines()[0][:60] for p in runs}
current = st.session_state.get("live_run", names[0])
name = picker.selectbox("Showing run", names, index=names.index(current) if current in names else 0,
                    format_func=lambda n: f"{n.split('_')[0]} · {first_line[n]}")
st.session_state.live_run = name
run_dir = live.RUNS / name

summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
doc_text = (run_dir / "document.txt").read_text(encoding="utf-8")
doc_graph = data.load_graph(run_dir / "document_triples.ttl")
base = ui.graph(list(data.SOURCES)[0])  # the repaired graph every run starts from
new_viol = pd.DataFrame(summary["new_violations"],
                        columns=["entity", "rule", "check", "property", "value", "message"])
facts = live.fact_rows(base, doc_graph, new_viol)
calls = live.agent_calls(run_dir)
graph_now = data.load_graph(run_dir / "graph_after.ttl")
questions = live.open_questions(graph_now, doc_graph)
decided = live.decisions(run_dir)
repairs = summary["repairs"]
v = summary["violations"]
now = len(json.loads((run_dir / "violations_after.json").read_text(encoding="utf-8")))

count = {k: sum(f["status"] == k for f in facts) for k in STATUS}
kept = sum(r["accepted"] for r in repairs)
states = ["warn" if summary["parse_error"] else "done", "done", "warn" if len(new_viol) else "done",
          "done" if repairs or not len(new_viol) else "skip", "you" if questions else "done"]
notes = [f"{summary['extract_seconds']} s",
         f"{len(facts)} facts · {count['conflict']} conflict",
         f"{len(new_viol)} new problem(s)",
         f"{kept} of {len(repairs)} fixed" if repairs else ("nothing to fix" if not len(new_viol) else "repair off"),
         f"{len(questions)} question(s)" if questions else f"{len(decided)} decision(s) made"]
draw_steps(slots, states, notes)

# ---------------------------------------------------------------- 4 · analytics for this run
ai_seconds = round(sum(c["seconds"] for c in calls), 1)
m = st.columns(4)
m[0].metric("Total time (s)", summary["seconds"], border=True)
m[1].metric("Time spent by AI agents", f"{ai_seconds / summary['seconds']:.0%}" if summary["seconds"] else "-",
            border=True, help=f"{ai_seconds} s in {len(calls)} AI calls; the rest is rules and code.")
m[2].metric("Facts in the document", len(facts), border=True,
            help=f"{count['new']} new, {count['already known']} already known, {count['conflict']} in conflict")
m[3].metric("Problems still open", now - v["base"], border=True,
            help="Rule problems this document caused that are still open (problems in the graph now, minus "
                 "the ones that were there before it).")

left, right = st.columns(2)
journey = ["After adding it", "After the Repair agent"] + (["After your decisions"] if decided else [])
open_counts = [v["after_merge"] - v["base"], v["final"] - v["base"]] + ([now - v["base"]] if decided else [])
left.plotly_chart(ui.bar(journey, open_counts, "Problems caused by this document, still open", fmt=".0f",
                         height=280), width="stretch")
if calls:
    labels = [f"{i + 1}. {c['agent']}: {c['task'][:40]}" for i, c in enumerate(calls)]
    right.plotly_chart(ui.bar(labels, [c["seconds"] for c in calls], "What the AI agents spent time on (s)",
                              horizontal=True, fmt=".1f", unit=" s", height=280), width="stretch")

# ---------------------------------------------------------------- 5 · step by step
st.divider()
st.subheader("📄 1 · Read: the Reader agent (AI)")
st.caption("The local LLM reads the document and fills a fixed form: machines, buffers, failures, maintenance "
           "actions, shifts and lines. It can only write into the form's fields.")
a, b = st.columns(2)
a.markdown("**The document**")
a.code(doc_text, language=None, wrap_lines=True)
b.markdown("**What the agent filled in**")
if summary["parse_error"]:
    b.error(f"The agent's answer did not fit the form: {summary['parse_error'][:300]}")
ext = json.loads((run_dir / "extraction.json").read_text(encoding="utf-8"))
b.json({k: [{f: x for f, x in item.items() if x is not None} for item in val]
        for k, val in ext.items() if val}, expanded=True)
for c in calls:
    if c["agent"] == "Reader agent":
        look_inside(c)

st.subheader("🧩 2 · Understand: the Organiser (code, no AI)")
st.caption("Plain code turns the form into facts in the factory's knowledge graph and compares each fact with "
           "what the factory already knows.")
st.dataframe(pd.DataFrame([{"": STATUS[f["status"]], "what": f["what"], "property": f["property"],
                            "this document says": f["value"], "our records say": f["our records say"]}
                           for f in facts]), hide_index=True, width="stretch")
with st.expander("Technical view: the RDF facts"):
    st.dataframe(pd.DataFrame({"fact": summary["triples"]}), hide_index=True, width="stretch")

st.subheader("🔎 3 · Check: the Rule checker (SHACL, no AI)")
st.caption("27 written rules (SHACL) check every fact, e.g. 'a machine has exactly one rated power'. "
           "The rules check form, not truth: a wrong value that breaks no rule passes.")
merged = live.merge(base, doc_graph)
if not len(new_viol):
    st.success("✓ The new facts break no rule.")
for _, r in new_viol.iterrows():
    vals = live.values_now(merged, r["entity"], r["property"])
    st.warning(f"**{live.name_of(merged, data.EX[r['entity']])}**: {r['message']}  \n"
               f"After adding this document it has: **{' · '.join(vals) if vals else 'nothing'}**", icon="⚠️")
with st.expander("Technical view: the SHACL results"):
    st.dataframe(new_viol, hide_index=True, width="stretch")

st.subheader("🛠️ 4 · Fix: the Repair agent (AI) and the safety gate (code)")
st.caption("For each problem the agent gets a multiple-choice question about only the broken property; all other "
           "facts are locked. Then a safety gate (code) tries each change on a copy and keeps it only if it "
           "lowers the number of rule problems on its own.")
repair_calls = [c for c in calls if c["agent"] == "Repair agent"]
if not repairs:
    st.info("Nothing to fix." if not len(new_viol) else "The Repair agent was switched off for this run.")
for rec in repairs:
    with st.container(border=True):
        g_name = live.name_of(merged, data.EX[rec["node"]])
        answer = rec.get("answer") or {}
        kept_p, rolled_p = rec.get("kept_properties", []), rec.get("rolled_back_properties", [])
        st.markdown(f"**{g_name}** · problems {rec.get('violations_before', '?')} → "
                    f"{rec.get('violations_after', '?')}")
        for prop, p in rec["plan"].items():
            label = live.PROP_LABEL.get(prop, prop)
            options = " · ".join(p["choices"]) if p["choices"] else "a free value"
            ans = answer.get(prop)
            if prop in kept_p or (rec["accepted"] and not kept_p and not rolled_p):
                gate = "✅ **kept**: it removed a problem"
            else:
                gate = "↩️ **rolled back**: it did not remove a problem on its own, so the facts stay as they were"
            st.markdown(f"- Question: which **{label}**? Options: {options}  \n"
                        f"  🤖 Agent answered: **{ans if ans is not None else 'none of these'}** → {gate}")
        if rec.get("reason") and "answer" not in rec:
            st.caption(rec["reason"])
        match = next((c for c in repair_calls if c["task"].startswith(f"Fix {rec['entity']}:")), None)
        if match:
            repair_calls.remove(match)
            look_inside(match)

st.subheader("👤 5 · You decide")
if questions:
    st.caption("The AI could not settle these, so every value is kept and flagged. You know the machine: choose "
               "the right one. Your choice is saved in this run's folder (decisions.json) and the AI's version is "
               "kept next to it.")
for q in questions:
    with st.container(border=True):
        st.markdown(f"**{q['what']} · {q['label']}: which value is right?**")
        st.caption(f"Rule: {q['problem']}")
        cols = st.columns(len(q["choices"]) + 1)
        for col, ch in zip(cols, q["choices"]):
            if col.button(f"{ch['value']} · {ch['source']}", key=f"{name}-{q['entity']}-{q['property']}-{ch['value']}",
                          disabled=readonly, width="stretch"):
                live.decide(run_dir, q["entity"], q["property"], ch["value"])
                ui.graph.clear()
                st.cache_data.clear()
                st.rerun()
        cols[-1].caption("Or leave it open: it stays flagged on the Data quality page.")
if not questions and not decided:
    st.success("✓ Nothing left for you to decide in this run.")
for d in decided:
    st.markdown(f"✅ You chose **{d['chosen']}** for {d['entity']} · "
                f"{live.PROP_LABEL.get(d['property'], d['property'])} (options {', '.join(d['choices'])}); "
                f"rule problems {d['violations_before']} → {d['violations_after']}.")

st.info(f"Everything is saved in `dashboard_runs/{name}/`: document, form, facts, rule results, repairs, your "
        "decisions and every AI call. Pick this run as the knowledge graph in the sidebar of any page to see its "
        "analytics.")
