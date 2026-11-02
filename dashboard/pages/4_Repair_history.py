"""Repair history: how the SHACL repair loop changed the graph, round by round,
for both designs - and every single repair attempt with the gate's decision."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, ui  # noqa: E402

ui.setup("Repair history", "🔧")
st.caption("Phase 3: violations go back to the local LLM for repair, at most 3 rounds. Two designs were run: "
           "**v1** let the LLM rewrite a whole entity; **v2** lets it change only the violated properties, as "
           "multiple choice where possible, and keeps a change only if the violation count drops (the gate). "
           "Sources: `results/phase3_v1_repair.json`, `results/phase3_v2_repair.json`.")

rounds = data.repair_rounds()
v2 = data.load_json("phase3_v2_repair.json")
c = st.columns(4)
c[0].metric("Repair attempts (v2)", v2["attempts"], border=True)
c[1].metric("Kept by the gate", v2["accepted"], border=True)
c[2].metric("Rolled back", v2["attempts"] - v2["accepted"], border=True,
            help="The change did not lower the number of violations, so the graph was left as it was.")
c[3].metric("Entities deleted", len(v2["deleted_entities"]), border=True,
            help="Deleted: " + (", ".join(v2["deleted_entities"]) or "none"))

# ---------------------------------------------------------------- round by round
left, right = st.columns(2)
with left:
    fig = ui.lines(rounds, "round", "violations", "design", "Rule violations after each round", fmt=".0f")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
with right:
    fig = ui.lines(rounds, "round", "f1", "design", "Correct facts (F1) after each round")
    st.plotly_chart(fig, width="stretch")
st.caption("Round 0 is the extracted graph before any repair. v1 made the graph worse (more violations, lower F1); "
           "v2 improved it a little and stopped when no repair passed the gate.")

# ---------------------------------------------------------------- per property
st.subheader("Which properties improved")
pp = data.per_predicate_f1()
only_changed = st.toggle("Only properties whose F1 changed", value=True)
if only_changed:
    pp = pp[pp["before repair"] != pp["after v2 repair"]]
fig = go.Figure()
for i, col in enumerate(["before repair", "after v2 repair"]):
    fig.add_trace(go.Bar(y=pp["property"], x=pp[col], name=col, orientation="h", marker_color=ui.SERIES[i],
                         text=[f"{v:.2f}" for v in pp[col]], textposition="outside",
                         hovertemplate="%{y}: %{x:.3f}<extra>" + col + "</extra>"))
fig.update_layout(height=max(220, 46 * len(pp) + 80), barmode="group", bargap=0.35, bargroupgap=0.1,
                  barcornerradius=4, margin=dict(l=8, r=8, t=8, b=8), font_family=ui.FONT,
                  legend=dict(orientation="h", y=-0.15, x=0))
fig.update_xaxes(range=[0, 1.12], title="F1")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- every attempt
st.subheader("Every v2 repair attempt")
att = data.repair_attempts_v2()
c1, c2 = st.columns(2)
rnd = c1.multiselect("Round", sorted(att["round"].unique()), default=sorted(att["round"].unique()))
sts = c2.multiselect("Gate decision", ["✓ kept", "↩ rolled back"], default=["✓ kept", "↩ rolled back"])
st.dataframe(att[att["round"].isin(rnd) & att["status"].isin(sts)], hide_index=True, width="stretch")
st.caption("“(free value)” means the property was missing, so there was nothing to choose from and the model had to "
           "write a value. That is where it sometimes invented one, and the gate still accepted it.")

# ---------------------------------------------------------------- v1 example
st.subheader("Why v1 failed: one entity before and after")
ex = data.repair_examples_v1()
names = [f"round {r['round']} · {r['entity']}" for r in ex]
pick = st.selectbox("Repair", names, index=next((i for i, r in enumerate(ex) if r["entity"] == "P2"), 0))
r = ex[names.index(pick)]
def grouped(facts):
    """Property -> all its values joined, so two labels stay visible as two labels."""
    out = {}
    for k, v in facts:
        out[k] = f"{out[k]} | {v}" if k in out else v
    return out


before, after = grouped(r["facts_before"]), grouped(r.get("facts_after") or [])
st.markdown("**Problem sent to the LLM:** " + " ".join(r["violations"]))
rows = []
for k in sorted(set(before) | set(after)):
    b, a = before.get(k, "—"), after.get(k, "—")
    rows.append({"property": k, "before": b, "after": a,
                 "change": "" if b == a else ("⚠ removed" if a == "—" else "⚠ changed")})
st.dataframe(rows, hide_index=True, width="stretch")
st.caption("Only the shown problem had to be fixed. Every ⚠ on a property that was not part of the problem is "
           "collateral damage from rewriting the whole entity.")
