"""Data quality: which SHACL rules the graph breaks, where, and - by comparing with
the ground truth - which wrong facts the rules cannot see at all."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard import ui  # noqa: E402

ui.setup("Data quality", "🛡️")
source = ui.source_picker()
v = ui.violations(source)
st.caption(f"SHACL validation of: **{source}** against `schema/shapes.ttl` "
           "(26 property shapes + 1 SPARQL topology rule).")

# ---------------------------------------------------------------- rule check summary
c = st.columns(3)
c[0].metric("Rule violations", len(v), border=True)
c[1].metric("Entities affected", v["entity"].nunique(), border=True)
c[2].metric("Different rules broken", v["rule"].nunique(), border=True)

if v.empty:
    st.success("✓ The graph conforms to every rule.")
else:
    left, right = st.columns([3, 2])
    with left:
        by_rule = v["rule"].value_counts()
        fig = ui.bar(by_rule.index.tolist(), by_rule.values.tolist(), "Violations per rule", horizontal=True,
                     fmt=".0f", height=max(260, 28 * len(by_rule) + 60))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**What the checks mean**")
        st.markdown("- **MaxCount**: too many values (e.g. two names)\n"
                    "- **MinCount**: a required value is missing\n"
                    "- **Class**: points to the wrong kind of thing\n"
                    "- **Datatype / MinExclusive**: wrong format or impossible value\n"
                    "- **SPARQL**: line layout broken (a machine feeds no buffer)")
        by_check = v["check"].value_counts()
        st.dataframe(by_check.rename("violations"), width="stretch")

    st.subheader("Violations by entity")
    entity = st.selectbox("Show the details of", ["(all)"] + sorted(v["entity"].unique()))
    shown = v if entity == "(all)" else v[v["entity"] == entity]
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"message": st.column_config.TextColumn("message", width="large")})

# ---------------------------------------------------------------- blind spots
st.subheader("What the rules cannot see")
if "Ground" in source:
    st.info("This is the ground truth itself; pick the extracted or repaired graph to compare it with the truth.")
else:
    q = ui.quality(source)
    s = q["score"]
    st.caption("Only possible here because a hand-written ground truth exists. In a real factory you would not "
               "have one, which is exactly why these silent errors matter.")
    c = st.columns(4)
    c[0].metric("Correct facts", f"{s['correct']} of {s['gold']}", border=True,
                help="Facts that match the ground truth exactly.")
    c[1].metric("Wrong facts flagged ✓", len(q["caught"]), border=True,
                help="A rule fires on exactly this property of this entity.")
    c[2].metric("Wrong facts not flagged ⚠", len(q["silent"]), border=True,
                help="Wrong or invented facts that break no rule, so SHACL stays silent.")
    c[3].metric("Missing facts", len(q["missing"]), border=True,
                help="In the ground truth, but not in this graph.")
    share = len(q["silent"]) / max(1, len(q["caught"]) + len(q["silent"]))
    st.warning(f"⚠ {share:.0%} of the wrong facts in this graph break no rule. SHACL checks whether the graph is "
               "well-formed, not whether each fact is true. Typical case: `machine_I1 mttrH 45.0` (the document "
               "says 45 min = 0.75 h).", icon="⚠️")
    left, right = st.columns(2)
    with left:
        with st.expander(f"Wrong facts SHACL did not flag ({len(q['silent'])})"):
            st.dataframe({"fact": q["silent"]}, hide_index=True, width="stretch")
    with right:
        with st.expander(f"Missing facts ({len(q['missing'])})"):
            st.dataframe({"fact": q["missing"]}, hide_index=True, width="stretch")
