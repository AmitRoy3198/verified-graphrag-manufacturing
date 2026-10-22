"""Triple-level precision / recall / F1 of an extracted graph against the ground truth.

A triple counts as correct only if subject, predicate and object all match.
Objects are compared after light normalisation:
  numbers by value (40 == 40.0), text case-insensitive and trimmed, IRIs exactly.
"""
from collections import defaultdict

from rdflib import Graph, Literal
from rdflib.namespace import XSD

NUMERIC = {XSD.decimal, XSD.integer, XSD.double, XSD.float, XSD.int}


def canon(o):
    if isinstance(o, Literal):
        if o.datatype in NUMERIC:
            return ("num", round(float(o), 3))
        if o.datatype == XSD.date:
            return ("date", str(o))
        return ("text", str(o).strip().lower())
    return ("iri", str(o))


def triple_keys(g: Graph) -> set:
    return {(str(s), str(p), canon(o)) for s, p, o in g}


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "correct": tp, "extracted": n_pred, "gold": n_gold}


def short(uri: str) -> str:
    return uri.split("#")[-1].split("/")[-1]


def score(pred: Graph, gold: Graph) -> dict:
    P, G = triple_keys(pred), triple_keys(gold)
    overall = prf(len(P & G), len(P), len(G))

    by_pred = defaultdict(lambda: [set(), set()])
    for t in P:
        by_pred[short(t[1])][0].add(t)
    for t in G:
        by_pred[short(t[1])][1].add(t)
    per_predicate = {k: prf(len(p & g), len(p), len(g)) for k, (p, g) in sorted(by_pred.items())}

    def fmt(t):
        return f"{short(t[0])} {short(t[1])} {t[2][1]}"

    return {
        "overall": overall,
        "per_predicate": per_predicate,
        "missing": sorted(fmt(t) for t in G - P),  # in ground truth, not extracted
        "wrong_or_extra": sorted(fmt(t) for t in P - G),  # extracted, not in ground truth
    }
