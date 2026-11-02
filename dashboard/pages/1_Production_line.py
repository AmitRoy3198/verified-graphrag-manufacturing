"""Production line: the material flow as stored in the graph, coloured by data health,
plus an inspector for any single entity (its facts, sources and rule violations)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from rdflib import RDF, URIRef  # noqa: E402

from dashboard import data, ui  # noqa: E402
from src.repair import code_of  # noqa: E402
from src.validate import MFG  # noqa: E402

ui.setup("Production line", "🔗")
source = ui.source_picker()
g = ui.graph(source)
issues = ui.violations(source)["entity"].value_counts().to_dict()
mt = ui.machines(source).set_index("Machine")


def fmt(x, unit):
    return "?" if x is None or pd.isna(x) else f"{x:g} {unit}"


PER_ROW = 6  # stations per row before the line wraps, so labels stay readable


def hover(name: str) -> str:
    """Everything the operator wants to know about one station, for the tooltip."""
    code, n = code_of(data.EX[name]), issues.get(name, 0)
    if name.startswith("machine_") and code in mt.index:
        r = mt.loc[code]
        txt = (f"<b>{code}</b> {r['Name'] if isinstance(r['Name'], str) else '(name unclear)'}<br>"
               f"cycle {fmt(r['Cycle time (s)'], 's')} · power {fmt(r['Power (kW)'], 'kW')}<br>"
               f"MTBF {fmt(r['MTBF (h)'], 'h')} · MTTR {fmt(r['MTTR (h)'], 'h')}")
        if r["Conflicts"]:
            txt += f"<br>conflict: {r['Conflicts']}"
    else:
        cap = ", ".join(str(o) for o in g.objects(data.EX[name], MFG.capacity)) or "?"
        txt = f"<b>{code}</b> buffer · capacity {cap}"
    return txt + f"<br>{ui.status(n)}"


def line_figure() -> go.Figure:
    """Stations laid out left to right, wrapping into rows. Machines are large
    squares, buffers small. Border colour AND the text under each node carry health."""
    seq, extras = data.flow_sequence(g)
    pos = {name: (i % PER_ROW, -(i // PER_ROW)) for i, name in enumerate(seq)}
    extra_row = -(len(seq) // PER_ROW) - 1.4
    pos |= {name: (i % PER_ROW, extra_row - i // PER_ROW) for i, name in enumerate(extras)}

    fig = go.Figure()
    for s, o in data.flow_edges(g):          # arrows for every 'feeds' fact in the graph
        if s in pos and o in pos:
            on_line = s in seq and o in seq and seq.index(o) - seq.index(s) == 1  # forward flow only
            fig.add_annotation(x=pos[o][0], y=pos[o][1], ax=pos[s][0], ay=pos[s][1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                               arrowcolor=ui.MUTED if on_line else ui.CRITICAL, standoff=22, startstandoff=22,
                               opacity=1 if on_line else 0.7)
    names = list(pos)
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], mode="markers+text",
        marker=dict(symbol="square", size=[44 if n.startswith("machine_") else 26 for n in names],
                    color="#fcfcfb", line=dict(width=3, color=[ui.GOOD if issues.get(n, 0) == 0 else ui.CRITICAL
                                                                for n in names])),
        text=[ui.status(issues.get(n, 0)) for n in names],
        textposition="bottom center", textfont=dict(size=11), hovertext=[hover(n) for n in names],
        hoverinfo="text"))
    for name in names:                       # the code inside each box
        fig.add_annotation(x=pos[name][0], y=pos[name][1], text=code_of(data.EX[name]), showarrow=False,
                           font=dict(size=11 if name.startswith("machine_") else 9, color="#0b0b0b"))
    if extras:
        fig.add_annotation(x=-0.4, y=extra_row + 0.45, xanchor="left", showarrow=False,
                           text="Not on the line (duplicates or invented by the LLM):", font=dict(size=12))
    rows = -min(y for _, y in pos.values()) + 1
    fig.update_layout(height=int(140 * rows + 40), margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                      font_family=ui.FONT, xaxis=dict(visible=False, range=[-0.6, PER_ROW - 0.4]),
                      yaxis=dict(visible=False, range=[min(y for _, y in pos.values()) - 0.7, 0.5]))
    return fig


st.caption("Material flow as stored in the graph, read left to right and row by row. "
           "Green border = ✓ no rule broken; red border = ⚠ open issues. Red arrows are 'feeds' facts that are "
           "not part of the main flow. Hover over a station for its data.")
st.plotly_chart(line_figure(), width="stretch")

seq, extras = data.flow_sequence(g)
st.markdown("**Flow found in the graph:** " + " → ".join(code_of(data.EX[n]) for n in seq))
if extras:
    st.markdown("**Not connected to the flow:** " + ", ".join(code_of(data.EX[n]) for n in extras))
bn = data.bottleneck(ui.machines(source))
if bn:
    st.markdown(f"**Bottleneck** (longest cycle time): **{bn}**")

# ---------------------------------------------------------------- entity inspector
st.subheader("Inspect an entity")
entities = sorted({data.local(s) for s in g.subjects(RDF.type, None) if str(s).startswith(str(data.EX))})
default = entities.index("machine_P2") if "machine_P2" in entities else 0
choice = st.selectbox("Entity", entities, index=default)
node = data.EX[choice]

left, right = st.columns([3, 2])
with left:
    facts = [{"property": str(p).split("#")[-1], "value": code_of(o) if isinstance(o, URIRef) else str(o)}
             for p, o in g.predicate_objects(node) if p != RDF.type]
    incoming = [{"property": f"← {str(p).split('#')[-1]}", "value": code_of(s)}
                for s, p in g.subject_predicates(node)]
    st.markdown(f"**Facts about {choice}** ({len(facts)} stored, {len(incoming)} pointing to it)")
    st.dataframe(pd.DataFrame(sorted(facts, key=lambda r: r["property"]) + incoming),
                 hide_index=True, width="stretch")
with right:
    v = ui.violations(source)
    v = v[v["entity"] == choice]
    st.markdown(f"**Rule check:** {ui.status(len(v))}")
    for _, r in v.iterrows():
        st.error(f"**{r['rule']}** ({r['check']}): {r['message']}"
                 + (f" Value: `{r['value']}`" if r["value"] else ""), icon="⚠️")
    docs = ui.provenance().get(choice, [])
    st.markdown("**Extracted from these documents:**" if docs else
                "**Source documents:** none recorded (this entity was not produced by the extraction).")
    for d in docs:
        with st.expander(d):
            st.text((data.DOCS / f"{d}.txt").read_text(encoding="utf-8"))
