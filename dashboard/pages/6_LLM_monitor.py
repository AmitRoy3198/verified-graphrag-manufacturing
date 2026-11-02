"""LLM monitor: every call the pipeline made to the local model, how long it took,
whether it returned valid JSON, and the exact prompt inputs and output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, ui  # noqa: E402
from src.config import NUM_CTX, NUM_PREDICT, SEED, TEMPERATURE  # noqa: E402

ui.setup("LLM monitor", "🧠")
st.caption(ui.MODEL_LINE)
include_demo = st.sidebar.toggle("Include demo calls", value=False,
                                 help="Calls made by `python -m src.demo` are logged separately "
                                      "(results/demo_log.jsonl) and are not part of the results.")
log = data.llm_log(include_demo)

# ---------------------------------------------------------------- tiles

c = st.columns(2) + st.columns(2)
model_id = log["model"].iloc[0] if not log.empty else "-"
c[0].metric("Model", model_id.split(":")[0].replace("qwen", "Qwen"), border=True,
            help=f"Ollama model id: {model_id} (open source, Apache-2.0)")
c[1].metric("Calls", len(log), border=True)
c[2].metric("LLM time (h)", f"{log['seconds'].sum() / 3600:.2f}", border=True)
c[3].metric("Not valid JSON", int((~log["valid_json"]).sum()), border=True,
            help="Structured calls whose raw output could not be parsed as JSON (e.g. cut off at the output cap).")
st.caption(f"Settings for every call: temperature {TEMPERATURE}, seed {SEED}, context {NUM_CTX} tokens, "
           f"at most {NUM_PREDICT} output tokens, model reloaded per call (keep_alive=0) for reproducibility.")

# ---------------------------------------------------------------- per stage
stages = list(dict.fromkeys(sorted(log["stage"])))
per = log.groupby("stage").agg(calls=("seconds", "size"), median_s=("seconds", "median"),
                               max_s=("seconds", "max"), hours=("seconds", lambda s: s.sum() / 3600),
                               invalid_json=("valid_json", lambda s: int((~s).sum()))).reindex(stages)
left, right = st.columns(2)
with left:
    fig = ui.bar(stages, per["calls"].tolist(), "Calls per pipeline stage", horizontal=True, fmt=".0f",
                 height=300)
    fig.update_traces(text=[str(int(v)) for v in per["calls"]], textposition="outside")
    st.plotly_chart(fig, width="stretch")
with right:
    fig = go.Figure()
    for s in stages:
        fig.add_trace(go.Box(y=[s] * int(per.loc[s, "calls"]), x=log.loc[log["stage"] == s, "seconds"],
                             orientation="h", marker=dict(color=ui.BLUE, size=5), line=dict(color=ui.BLUE, width=1.5),
                             boxpoints="all", jitter=0.4, pointpos=0, name=s,
                             hovertemplate="%{x:.1f} s<extra>" + s + "</extra>"))
    fig.update_layout(title=dict(text="Seconds per call (log scale)", font_size=15), height=300, showlegend=False,
                      margin=dict(l=8, r=8, t=44, b=8), font_family=ui.FONT)
    fig.update_xaxes(type="log", dtick=1, title="seconds")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
st.dataframe(per.round(1), width="stretch")
slowest = log.loc[log["seconds"].idxmax()]
note = (" This is extraction run v1, where the model repeated one JSON object without end; it is why every call "
        "now has an output cap." if slowest["stage"].endswith("Extraction") and slowest["seconds"] > 3600 else "")
st.warning(f"⚠ Slowest call: {slowest['seconds']:.0f} s in {slowest['stage']} "
           f"({slowest['output chars']:,} characters of output).{note}", icon="⚠️")

# ---------------------------------------------------------------- call viewer
st.subheader("Look at one call")
c1, c2 = st.columns([1, 2])
stage = c1.selectbox("Stage", stages)
calls = log[log["stage"] == stage].reset_index(drop=True)
idx = c2.selectbox("Call", calls.index, format_func=lambda i: (
    f"#{i + 1} · {calls.loc[i, 'time']:%Y-%m-%d %H:%M} · {calls.loc[i, 'seconds']:.0f} s · {calls.loc[i, 'prompt']}"
    + ("" if calls.loc[i, "valid_json"] else " · ⚠ not valid JSON")))
call = calls.loc[idx]
left, right = st.columns(2)
with left:
    st.markdown("**Inputs filled into the prompt template**")
    st.caption(f"Template: `prompts/{call['prompt']}*.txt`")
    st.code(json.dumps(call["inputs"], indent=2, ensure_ascii=False)[:6000], language="json")
with right:
    st.markdown(f"**Raw model output** ({call['output chars']:,} characters)")
    out = call["output"]
    st.code(out if len(out) < 6000 else out[:6000] + "\n... (cut here; see results/llm_log.jsonl)",
            language="json" if call["valid_json"] and call["prompt"] != "hello" else "text")
