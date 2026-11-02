"""Verified Factory Knowledge - Overview.

    .venv\\Scripts\\python -m streamlit run dashboard\\Home.py --server.port 8503

Read-only: every number on these pages comes from results/ or from a knowledge
graph file. Nothing here calls the LLM or writes a file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from dashboard import data, ui  # noqa: E402

ui.setup("Verified Factory Knowledge", "🏭")
st.markdown("From shop-floor documents to a **checked knowledge graph** of the production line, "
            "and answers you can trace back to their source.")
st.caption(ui.MODEL_LINE)

o = data.overview_numbers()

# ---------------------------------------------------------------- headline tiles
c = st.columns(4)
c[0].metric("Documents read", o["documents"], border=True,
            help="Data sheets, maintenance reports, shift notes and audits in data/docs/.")
c[1].metric("Facts extracted", o["triples_extracted"], border=True,
            help=f"The hand-written ground truth has {o['triples_gold']} facts; the rest are duplicates, "
                 "conflicts or inventions.")
c[2].metric("Open rule violations", o["violations_after"], o["violations_after"] - o["violations_before"],
            delta_color="inverse", border=True,
            help=f"SHACL found {o['violations_before']} before repair and {o['violations_after']} after.")
c[3].metric("Correct facts (F1)", f"{o['f1_after']:.3f}", f"{o['f1_after'] - o['f1_before']:+.3f} from repair",
            border=True, help="Triple-level F1 against the ground truth (1.0 = perfect).")

# ---------------------------------------------------------------- pipeline strip
st.subheader("The pipeline, stage by stage")
stages = [
    ("1 · Read", "LLM fills a fixed JSON form per document",
     f"**{o['documents']}** documents → **{o['triples_extracted']}** facts"),
    ("2 · Organise", "Facts become an RDF knowledge graph",
     "6 classes, codes become IRIs (W1 → machine_W1)"),
    ("3 · Check", "SHACL rules test the graph",
     f"**{o['shapes']}** rules, {o['constraints']} constraints → **{o['violations_before']}** violations"),
    ("4 · Fix", "Targeted LLM repair, kept only if violations drop",
     f"{o['repairs_attempted']} attempts, **{o['repairs_accepted']}** kept → **{o['violations_after']}** left"),
    ("5 · Answer", "Questions answered from the graph (GraphRAG)",
     f"**{o['qa']['graphrag_repaired']:.0%}** correct (perfect graph: {o['qa']['graphrag_gold']:.0%})"),
]
for col, (name, what, result) in zip(st.columns(len(stages)), stages):
    with col.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(what)
        st.markdown(result)

# ---------------------------------------------------------------- needs attention
left, right = st.columns([3, 2])
with left:
    st.subheader("Needs attention")
    source = "Repaired graph (after SHACL repair)"
    v = ui.violations(source)
    st.caption(f"Entities in the **repaired** graph that still break a rule ({len(v)} violations). "
               "Open *Data quality* for the details.")
    if v.empty:
        st.success("✓ No open violations.")
    else:
        per = (v.groupby("entity").agg(issues=("rule", "size"), rules=("rule", lambda r: ", ".join(sorted(set(r)))))
               .reset_index().sort_values(["issues", "entity"], ascending=[False, True]))
        per.insert(0, "status", per["issues"].map(ui.status))
        per = per.drop(columns="issues")
        st.dataframe(per, hide_index=True, width="stretch", height=300)

with right:
    st.subheader("Answer accuracy")
    qa = o["qa"]
    labels = ["Vector RAG (raw documents)", "GraphRAG (repaired graph)", "GraphRAG (perfect graph)"]
    fig = ui.bar(labels, [qa["vector_rag"], qa["graphrag_repaired"], qa["graphrag_gold"]],
                 "Share of 25 questions answered correctly (lenient)", horizontal=True, fmt=".0%", height=260)
    fig.update_traces(text=[f"{x:.0%}" for x in [qa["vector_rag"], qa["graphrag_repaired"], qa["graphrag_gold"]]],
                      textposition="outside")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("With 25 questions no difference is statistically significant (sign test p ≥ 0.45).")

# ---------------------------------------------------------------- footer
c = st.columns(3)
c[0].metric("LLM calls logged", o["llm_calls"], border=True)
c[1].metric("LLM time on this laptop (h)", o["llm_hours"], border=True)
c[2].metric("Rules tested to fail on purpose", o["shapes"], border=True)
st.caption("Sources: `results/phase2_scores.json`, `phase3_v2_repair.json`, `phase4_scores.json`, "
           "`llm_usage.json`, `phase1_schema.json`.")
