"""Shared Streamlit helpers: page setup, the graph selector, caching, chart style."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402

# Colour roles (dataviz reference palette). Status colours always come with an icon + label.
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
MUTED = "#898781"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

MODEL_LINE = ("Local open-source LLM: **Qwen2.5-3B-Instruct** (Apache-2.0) via Ollama on this laptop · "
              "temperature 0 · no cloud API. No LLM computes a number shown on this dashboard.")


def setup(title: str, icon: str) -> None:
    page = title if title == "Verified Factory Knowledge" else f"{title} · Verified Factory Knowledge"
    st.set_page_config(page_title=page, page_icon=icon, layout="wide")
    st.title(title)


@st.cache_resource(show_spinner="Loading graph ...")
def graph(source: str):
    return data.load_graph(data.SOURCES[source])


@st.cache_resource(show_spinner=False)
def gold():
    return data.load_graph(data.DATA / "ground_truth.ttl")


@st.cache_data(show_spinner="Checking the graph with SHACL ...")
def violations(source: str):
    return data.violations_df(graph(source))


@st.cache_data(show_spinner=False)
def machines(source: str):
    return data.machine_table(graph(source))


@st.cache_data(show_spinner=False)
def query(source: str, name: str):
    return data.query(graph(source), name)


@st.cache_data(show_spinner=False)
def metrics(source: str):
    return data.headline_metrics(graph(source))


@st.cache_data(show_spinner="Comparing with the ground truth ...")
def quality(source: str):
    return data.silent_errors(graph(source), gold())


@st.cache_data(show_spinner=False)
def provenance():
    from src.repair import provenance as prov
    return {data.local(k): sorted(v) for k, v in prov().items()}


def source_picker(include_gold: bool = True) -> str:
    """One selector in the sidebar, shared by every page."""
    options = [s for s in data.SOURCES if include_gold or "Ground" not in s]
    current = st.session_state.get("source", options[0])
    choice = st.sidebar.radio("Knowledge graph", options, index=options.index(current) if current in options else 0,
                              help="Switch between the graph before repair, after repair, and the hand-written "
                                   "ground truth, to see how extraction errors change what the operator sees.")
    st.session_state.source = choice
    st.sidebar.caption("Every number comes from the graph (SPARQL) or from `results/`.")
    return choice


def status(n: int) -> str:
    return "✓ OK" if n == 0 else f"⚠ {n} issue{'s' if n != 1 else ''}"


def bar(x, y, title: str, *, horizontal=False, highlight=None, highlight_label=None, fmt=".2f",
        unit: str = "", height: int = 320) -> go.Figure:
    """A single-series bar chart. One hue; an optional highlighted bar gets
    slot 2 plus a direct text label, so colour never carries meaning alone."""
    colors = [ORANGE if highlight is not None and xi == highlight else BLUE for xi in x]
    texts = [highlight_label if highlight is not None and xi == highlight else "" for xi in x]
    kw = dict(x=y, y=x, orientation="h") if horizontal else dict(x=x, y=y)
    fig = go.Figure(go.Bar(**kw, marker_color=colors, text=texts, textposition="outside",
                           hovertemplate=(("%{y}: %{x:" if horizontal else "%{x}: %{y:") + fmt + "}" + unit
                                          + "<extra></extra>")))
    fig.update_layout(title=dict(text=title, font_size=15), height=height, margin=dict(l=8, r=8, t=44, b=8),
                      font_family=FONT, barcornerradius=4, bargap=0.45, showlegend=False)
    fig.update_xaxes(showgrid=horizontal)
    fig.update_yaxes(showgrid=not horizontal, autorange="reversed" if horizontal else None)
    return fig


def sparql(name: str) -> None:
    with st.expander("SPARQL behind this chart"):
        st.code(data.QUERIES[name], language="sparql")
