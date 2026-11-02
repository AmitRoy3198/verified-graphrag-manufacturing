"""Ask the factory: the 25-question evaluation (Phase 4 v2), answer by answer.
Live questions to the local model come in Phase D3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, ui  # noqa: E402

ui.setup("Ask the factory", "💬")
st.caption("25 questions with SPARQL-verified answers, each answered four ways by the local LLM. "
           "Source: `results/phase4_answers_*.json`. Asking your own question live arrives in Phase D3.")

qa = data.qa_answers()
conds = list(data.QA_CONDITIONS.values())
score = st.radio("Count an answer as right when", ["it matches exactly", "it matches leniently"], horizontal=True,
                 help="Exact: the bare value or name equals the answer. Lenient: numbers within 2 %, or the "
                      "expected code appears somewhere in the answer.")
key = "exact" if score.endswith("exactly") else "lenient"

# ---------------------------------------------------------------- accuracy by type
types = ["1-hop", "multi-hop", "aggregation"]
acc = qa.groupby(["condition", "type"])[key].mean().unstack().reindex(index=conds, columns=types)
for col, cond in zip(st.columns(2) + st.columns(2), conds):
    col.metric(cond, f"{qa[qa['condition'] == cond][key].mean():.0%}", border=True)

fig = go.Figure()
for i, cond in enumerate(conds):
    fig.add_trace(go.Bar(x=types, y=acc.loc[cond], name=cond, marker_color=ui.SERIES[i],
                         hovertemplate=cond + "<br>%{x}: %{y:.0%}<extra></extra>"))
fig.update_layout(title=dict(text="Share answered right, by question type", font_size=15), barmode="group",
                  height=340, bargap=0.3, bargroupgap=0.08, barcornerradius=4, font_family=ui.FONT,
                  margin=dict(l=8, r=8, t=44, b=8), legend=dict(orientation="h", y=-0.2, x=0))
fig.update_yaxes(range=[0, 1], tickformat=".0%")
st.plotly_chart(fig, width="stretch")
st.caption("10 one-hop, 8 multi-hop and 7 aggregation questions, so one question is 10-14 % of its group. "
           "No difference here is statistically significant.")

# ---------------------------------------------------------------- answer grid
st.subheader("Every answer")
grid = qa.pivot(index="id", columns="condition", values="result").reindex(columns=conds)
level = {"✓ right": 2, "~ right (lenient)": 1, "✗ wrong": 0}
z = grid.map(level.get)
answers = qa.pivot(index="id", columns="condition", values="answer").reindex(columns=conds)
expected = qa.drop_duplicates("id").set_index("id")["expected"]
hover = [[f"{q}: expected {expected[q]}<br>{c}: {str(answers.loc[q, c])[:60]!r}<br>{grid.loc[q, c]}"
          for c in conds] for q in grid.index]
fig = go.Figure(go.Heatmap(
    z=z.values, x=conds, y=grid.index, text=grid.map(lambda s: s.split()[0]).values, texttemplate="%{text}",
    hovertext=hover, hoverinfo="text", showscale=False, xgap=2, ygap=2, zmin=0, zmax=2,
    colorscale=[[0, ui.CRITICAL], [0.5, ui.WARNING], [1, ui.GOOD]]))
fig.update_layout(height=26 * len(grid) + 80, margin=dict(l=8, r=8, t=8, b=8), font_family=ui.FONT)
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(side="top")
st.plotly_chart(fig, width="stretch")
st.caption("✓ right (exact) · ~ right only with the lenient score · ✗ wrong. Hover for the answer.")

# ---------------------------------------------------------------- one question
st.subheader("One question in detail")
questions = qa.drop_duplicates("id").set_index("id")
qid = st.selectbox("Question", questions.index, format_func=lambda i: f"{i} · {questions.loc[i, 'question']}")
q = questions.loc[qid]
st.markdown(f"**{q['question']}**  \nType: {q['type']} · Expected answer: **{q['expected']}**")
for col, cond in zip(st.columns(2) + st.columns(2), conds):
    r = qa[(qa["id"] == qid) & (qa["condition"] == cond)].iloc[0]
    with col.container(border=True):
        st.markdown(f"**{cond}** · {r['result']}")
        st.markdown(f"Answer: `{r['answer'] or '(empty)'}`")
        if cond == "Vector RAG":
            st.caption(f"Retrieved documents: {', '.join(r['retrieved_docs'])} · needed-document recall "
                       f"{r['doc_recall']:.0%}")
        else:
            st.caption(f"Linked entities: {', '.join(r['seeds']) or 'none'} · {int(r['facts'])} fact lines in the context")
        with st.expander("Evidence the model cited"):
            for e in r["evidence"] or ["(none)"]:
                st.text(e)

# ---------------------------------------------------------------- paired test
st.subheader("Is any difference real?")
paired = data.load_json("phase4_scores.json")["paired_comparisons_lenient"]
rows = []
for pair, v in paired.items():
    a, b = pair.split(" vs ")
    rows.append({"comparison": f"{data.QA_CONDITIONS[a]} vs {data.QA_CONDITIONS[b]}", "both right": v["both_right"],
                 "only first right": v[f"only_{a}"], "only second right": v[f"only_{b}"],
                 "both wrong": v["both_wrong"], "sign test p": v["sign_test_p"]})
st.dataframe(rows, hide_index=True, width="stretch")
st.caption("A sign test on the questions where two methods disagree (lenient score). p ≥ 0.05 everywhere: "
           "with 25 questions these are indications, not proof.")
