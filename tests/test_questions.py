"""Phase 4: every expected answer in data/questions.json is checked against the
ground-truth graph with its SPARQL query, and every listed source document exists."""
import json

import pytest
from rdflib import Graph, Literal

from src.config import DATA, DOCS

QUESTIONS = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
GOLD = Graph().parse(DATA / "ground_truth.ttl")
PREFIXES = """PREFIX ex: <http://example.org/line/>
PREFIX mfg: <http://example.org/mfg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def test_question_set_shape():
    assert len(QUESTIONS) == 25
    assert len({q["id"] for q in QUESTIONS}) == 25
    assert {q["type"] for q in QUESTIONS} == {"1-hop", "multi-hop", "aggregation"}


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_answer_matches_ground_truth(q):
    rows = list(GOLD.query(PREFIXES + q["sparql"]))
    assert len(rows) == 1, f"{q['id']}: expected one result, got {rows}"
    value = rows[0][0]
    if q["kind"] == "number":
        assert float(value) == pytest.approx(q["answer"])
    elif q["kind"] == "text":
        assert isinstance(value, Literal) and str(value) == q["answer"]
    else:  # entity: the IRI ends with the code, e.g. .../machine_K1
        assert str(value).rsplit("_", 1)[-1] == q["answer"].replace("-", "").upper()


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_source_documents_exist(q):
    for d in q["docs"]:
        assert (DOCS / f"{d}.txt").exists(), d
