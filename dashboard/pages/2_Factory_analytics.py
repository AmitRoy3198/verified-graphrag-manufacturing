"""Factory analytics: the operator's numbers, computed from the graph with SPARQL
(never by the LLM), and the same numbers side by side on all three graphs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, ui  # noqa: E402

ui.setup("Factory analytics", "📊")
source = ui.source_picker()
mt = ui.machines(source)
m = ui.metrics(source)
st.caption(f"Computed from: **{source}**. Values the graph holds twice are not averaged or guessed: "
           "they are left empty and listed under *Conflicts*.")

# ---------------------------------------------------------------- headline tiles
c = st.columns(3)
c[0].metric("Bottleneck", m["Bottleneck (longest cycle)"] or "unknown", border=True,
            help="Machine with the longest cycle time; it limits the line's output.")
c[1].metric("Most failures", m["Machine with most failures"] or "unknown", border=True)
c[2].metric("Conflicting machines", m["Machines with conflicting data"], border=True,
            help="Machines where the graph holds two different values for one property.")
c = st.columns(3)
c[0].metric("Failures recorded", m["Failures"], border=True)
c[1].metric("Downtime (h)", f"{m['Total downtime (h)']:g}", border=True)
c[2].metric("Night-shift failures", m["Night-shift failures"], border=True)

# ---------------------------------------------------------------- machines
st.subheader("Machines")
st.dataframe(
    mt, hide_index=True, width="stretch",
    column_config={
        "Availability": st.column_config.ProgressColumn(
            "Availability", help="MTBF / (MTBF + MTTR): share of time the machine is not under repair.",
            min_value=0.0, max_value=1.0, format="percent"),
        "Conflicts": st.column_config.TextColumn("Conflicts", width="large"),
    })

known = mt.dropna(subset=["Cycle time (s)"])
left, right = st.columns(2)
with left:
    fig = ui.bar(known["Machine"].tolist(), known["Cycle time (s)"].tolist(), "Cycle time per machine (s)",
                 highlight=m["Bottleneck (longest cycle)"], highlight_label="bottleneck", fmt=".0f", unit=" s")
    st.plotly_chart(fig, width="stretch")
with right:
    av = mt.dropna(subset=["Availability"])
    fig = ui.bar(av["Machine"].tolist(), av["Availability"].tolist(), "Availability per machine",
                 fmt=".1%")
    fig.update_yaxes(tickformat=".0%", range=[min(0.9, av["Availability"].min() - 0.01) if not av.empty else 0, 1])
    st.plotly_chart(fig, width="stretch")
st.caption("Bars follow the material flow (left = start of the line). Machines with a missing or conflicting "
           "value are left out of the chart and flagged in the table.")

# ---------------------------------------------------------------- failures
st.subheader("Failures and maintenance")
dt = ui.query(source, "downtime_by_machine")
shifts = ui.query(source, "failures_by_shift")
actions = ui.query(source, "actions_by_type")
left, right = st.columns(2)
with left:
    order = {code: i for i, code in enumerate(data.line_order(ui.graph(source)))}
    d = dt.assign(pos=dt["machine"].map(order).fillna(99)).sort_values("pos")
    fig = ui.bar(d["machine"].tolist(), d["downtime_h"].tolist(), "Production downtime per machine (h)",
                 highlight=m["Machine with most failures"], highlight_label="most failures", fmt=".2f", unit=" h")
    st.plotly_chart(fig, width="stretch")
    ui.sparql("downtime_by_machine")
with right:
    s = shifts.assign(shift=shifts["shift"].str.capitalize())
    s = s.set_index("shift").reindex([x for x in ["Early", "Late", "Night", "Unknown"] if x in set(s["shift"])]) \
        .reset_index()
    fig = ui.bar(s["shift"].tolist(), s["failures"].tolist(), "Failures per shift", fmt=".0f")
    st.plotly_chart(fig, width="stretch")
    ui.sparql("failures_by_shift")

c = st.columns(len(actions) if len(actions) else 1)
for col, (_, r) in zip(c, actions.iterrows()):
    col.metric(f"{r['type']} maintenance actions", int(r["actions"]), f"{r['hours']:.2f} h of work",
               delta_color="off", border=True)
ui.sparql("actions_by_type")

st.markdown("**Failure log**")
st.dataframe(data.failure_log(ui.graph(source)), hide_index=True, width="stretch")

# ---------------------------------------------------------------- comparison
st.subheader("Same analytics on all three graphs")
st.caption("How extraction errors change what an operator would see. ⚠ marks a value that differs from the "
           "hand-written ground truth.")
gold_name = next(n for n in data.SOURCES if "Ground" in n)
table = pd.DataFrame({name.split(" (")[0]: ui.metrics(name) for name in data.SOURCES})
gold_col = gold_name.split(" (")[0]
shown = table.astype(str)
for col in table.columns:
    if col != gold_col:
        shown[col] = [f"⚠ {v}" if str(v) != str(g) else str(v) for v, g in zip(table[col], table[gold_col])]
st.dataframe(shown, width="stretch")
