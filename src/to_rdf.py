"""Deterministic conversion: LLM JSON (Extraction) -> RDF triples.

No LLM here. Two jobs:
1. Turn a free-text code into a stable IRI: "Press P2" -> ex:machine_P2.
2. Turn each JSON field into one triple with the ontology property.
If the LLM gives a wrong kind of value (e.g. a date "12 March 2026") we
keep it as plain text on purpose, so SHACL can report it.
"""
import re
from datetime import date

from rdflib import Graph, Literal, RDF, RDFS
from rdflib.namespace import XSD

from src.extract_schema import Extraction
from src.validate import EX, MFG

CODE = re.compile(r"[A-Za-z]+-?\d+")  # P2, B3, INC-205, WO-1101


def norm_id(text: str) -> str:
    """'Press P2' -> 'P2', 'INC-205' -> 'INC205', 'Night shift' -> 'NIGHT', 'Line A' -> 'LINEA'."""
    text = text.strip()
    m = CODE.search(text)
    if m:
        text = m.group()
    else:
        text = re.sub(r"\bshift\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def iri(kind: str, text: str):
    return EX[f"{kind}_{norm_id(text)}"]


def dec(x: float) -> Literal:
    return Literal(str(float(x)), datatype=XSD.decimal)


def whole(x: float) -> Literal:
    """Integer if the number is whole (40.0 -> 40), else keep the decimal so SHACL can flag it."""
    return Literal(int(x)) if float(x).is_integer() else dec(x)


def date_lit(text: str) -> Literal:
    try:
        return Literal(date.fromisoformat(text.strip()))  # xsd:date
    except ValueError:
        return Literal(text)  # wrong format stays a string -> SHACL datatype violation


def extraction_to_graph(ex: Extraction) -> Graph:
    g = Graph()
    g.bind("ex", EX)
    g.bind("mfg", MFG)

    def add(s, p, value, convert=None):
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        g.add((s, p, convert(value) if convert else Literal(value.strip() if isinstance(value, str) else value)))

    for m in ex.machines:
        s = iri("machine", m.id)
        g.add((s, RDF.type, MFG.Machine))
        add(s, RDFS.label, m.name)
        add(s, MFG.machineType, m.machine_type)
        add(s, MFG.cycleTimeS, m.cycle_time_s, dec)
        add(s, MFG.powerKW, m.power_kw, dec)
        add(s, MFG.mtbfH, m.mtbf_h, dec)
        add(s, MFG.mttrH, m.mttr_h, dec)
        add(s, MFG.partOfLine, m.line, lambda v: iri("line", v))
        add(s, MFG.feeds, m.feeds_buffer, lambda v: iri("buffer", v))

    for b in ex.buffers:
        s = iri("buffer", b.id)
        g.add((s, RDF.type, MFG.Buffer))
        add(s, RDFS.label, b.name)
        add(s, MFG.capacity, b.capacity, whole)
        add(s, MFG.feeds, b.feeds_machine, lambda v: iri("machine", v))
        if b.fed_by_machine:
            g.add((iri("machine", b.fed_by_machine), MFG.feeds, s))

    for f in ex.failures:
        s = iri("failure", f.id)
        g.add((s, RDF.type, MFG.Failure))
        add(s, MFG.onMachine, f.machine, lambda v: iri("machine", v))
        add(s, MFG.failureMode, f.failure_mode)
        add(s, MFG.downtimeH, f.downtime_h, dec)
        add(s, MFG.date, f.date, date_lit)
        add(s, MFG.duringShift, f.shift, lambda v: iri("shift", v))

    for a in ex.maintenance_actions:
        s = iri("action", a.id)
        g.add((s, RDF.type, MFG.MaintenanceAction))
        add(s, MFG.onMachine, a.machine, lambda v: iri("machine", v))
        add(s, MFG.actionType, a.action_type)
        add(s, MFG.date, a.date, date_lit)
        add(s, MFG.durationH, a.duration_h, dec)
        add(s, MFG.addresses, a.addresses_failure, lambda v: iri("failure", v))

    for sh in ex.shifts:
        s = iri("shift", sh.id)
        g.add((s, RDF.type, MFG.Shift))
        add(s, RDFS.label, sh.name)
        add(s, MFG.startHour, sh.start_hour, lambda v: Literal(int(v)))

    for ln in ex.lines:
        s = iri("line", ln.id)
        g.add((s, RDF.type, MFG.ProductionLine))
        add(s, RDFS.label, ln.name)
        add(s, MFG.finalMachine, ln.final_machine, lambda v: iri("machine", v))

    return g
