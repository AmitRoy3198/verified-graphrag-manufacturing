"""Guided demo: one example through every stage of the pipeline, printed step by step.

LLM calls made here are logged to results/demo_log.jsonl (not llm_log.jsonl),
so the logged project results stay unchanged.

Run:
  python -m src.demo 1                    Ollama + LangChain: one call
  python -m src.demo 2 [document]         LLM extraction of one document -> JSON -> RDF -> scored
  python -m src.demo 3                    SHACL validation of the extracted graph
  python -m src.demo 4 [entity]           one targeted + gated repair
  python -m src.demo 5 ["question"]       the same question with vector RAG and GraphRAG
"""
import json
import sys
import time
import urllib.request

from rdflib import Graph

import src.llm as llm
from src.config import DATA, DOCS, MODEL, NUM_CTX, RESULTS, SEED, TEMPERATURE, load_prompt

llm.LOG_FILE = RESULTS / "demo_log.jsonl"  # keep demo calls out of the project log


def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def short(g: Graph, t):
    return " ".join(g.namespace_manager.normalizeUri(x) if not hasattr(x, "datatype") else repr(str(x)) for x in t)


# ---------------------------------------------------------------- step 1
def step1():
    from langchain_core.prompts import ChatPromptTemplate

    title("1a. Is the Ollama server running? (HTTP GET localhost:11434/api/tags)")
    tags = json.load(urllib.request.urlopen("http://localhost:11434/api/tags"))
    for m in tags["models"]:
        print(f"  installed model: {m['name']}  ({m['size'] / 1e9:.1f} GB)")
    print(f"  settings used: model={MODEL} temperature={TEMPERATURE} seed={SEED} num_ctx={NUM_CTX} keep_alive=0")

    title("1b. A LangChain chain = prompt template | model")
    prompt = ChatPromptTemplate.from_template(load_prompt("hello"))
    print("  prompt template (prompts/hello.txt):", repr(load_prompt("hello")))
    chain = prompt | llm.get_llm()
    q = "What does a buffer do between two machines in a production line?"
    print("  filled prompt:", prompt.format(question=q))
    t = time.perf_counter()
    answer = llm.run_logged(chain, {"question": q}, phase="demo", prompt_name="hello")
    print(f"\n  model answer ({time.perf_counter() - t:.1f} s, includes loading the model into RAM):\n  {answer}")


# ---------------------------------------------------------------- step 2
def step2(doc="13_report_WO-1104"):
    from src.extract import build_chain, extract_doc
    from src.score import score
    from src.to_rdf import extraction_to_graph

    path = DOCS / f"{doc}.txt"
    title(f"2a. The input document: data/docs/{path.name}")
    print(path.read_text(encoding="utf-8"))

    title("2b. LLM extraction (prompts/extract_system.txt + JSON schema from src/extract_schema.py)")
    print("  calling the model ... (about 1 minute on this laptop)")
    ext, info = extract_doc(build_chain(), path)
    print(f"  done in {info['seconds']} s, parse error: {info['parse_error']}")
    print("  JSON returned (empty lists and null fields hidden):")
    print(json.dumps(ext.model_dump(exclude_none=True, exclude_defaults=True), indent=2))

    title("2c. JSON -> RDF (src/to_rdf.py, no LLM)")
    g = extraction_to_graph(ext)
    for t in sorted(g):
        print("  ", short(g, t))

    title("2d. Score against the ground truth, only for the entities in this document")
    gold = Graph().parse(DATA / "ground_truth.ttl")
    subjects = set(g.subjects())
    gold_part = Graph()
    for t in gold:
        if t[0] in subjects:
            gold_part.add(t)
    s = score(g, gold_part)
    print("  ", s["overall"])
    print("   missing (in ground truth, not extracted):", s["missing"] or "none")
    print("   wrong or extra (extracted, not in ground truth):", s["wrong_or_extra"] or "none")


# ---------------------------------------------------------------- step 3
def step3(entity="machine_P2"):
    from collections import Counter

    from src.validate import validate_graph

    g = Graph().parse(RESULTS / "phase2_extracted.ttl")
    title("3a. Validate the full extracted graph (results/phase2_extracted.ttl) with pySHACL")
    conforms, violations = validate_graph(g)
    print(f"  triples: {len(g)}   conforms: {conforms}   violations: {len(violations)}")
    for k, n in Counter(f"{v['shape']} / {v['component']}" for v in violations).most_common(8):
        print(f"   {n:3}  {k}")

    title(f"3b. The violations of one entity: ex:{entity}")
    for v in violations:
        if v["focus"] == f"ex:{entity}":
            print(f"   [{v['component']}] path={v['path']} value={v['value'] or '-'}\n      message: {v['message']}")
    print(f"\n  its facts in the extracted graph:")
    for t in sorted(g.triples((g.namespace_manager.expand_curie(f"ex:{entity}"), None, None))):
        print("   ", short(g, t))

    title("3c. What SHACL cannot see: a wrong value that breaks no rule")
    i1 = g.namespace_manager.expand_curie("ex:machine_I1")
    mttr = [str(o) for o in g.objects(i1, g.namespace_manager.expand_curie("mfg:mttrH"))]
    print(f"  machine_I1 mttrH = {mttr}  (document says 45 min = 0.75 h)")
    print("  45 > 0 and 45 < mtbf 600, so every rule is satisfied -> no violation is reported.")


# ---------------------------------------------------------------- step 4
def step4(entity="machine_P2"):
    from src import repair_v2
    from src.repair import provenance
    from src.validate import EX, validate_graph

    g = Graph().parse(RESULTS / "phase2_extracted.ttl")
    node = EX[entity]
    repair_v2.SOURCES.update(provenance())
    current = [v for v in validate_graph(g)[1] if v["focus"] == f"ex:{entity}"]
    cls = repair_v2.node_class(g, node)
    plan = repair_v2.plan_fix(g, node, cls, current)

    title(f"4a. Which properties of {entity} may change? (only the violated ones)")
    for prop, entry in plan.items():
        print(f"  {prop}: choices={entry['choices']}  problems={entry['problems']}")
    model = repair_v2.fix_model(plan)
    print("\n  JSON schema built just for this call (the model can only fill these fields):")
    print(json.dumps(model.model_json_schema()["properties"], indent=2))

    title("4b. The prompt the model gets (prompts/repair_v2_*.txt)")
    inputs = repair_v2.build_inputs(g, node, cls, plan)
    print(load_prompt("repair_v2_user").format(**inputs)[:2500])

    title("4c. Ask the model, apply the fix to a copy, keep it only if total violations drop")
    print("  calling the model ...")
    rec = repair_v2.gated_repair(g, node, current, repair_v2.make_llm_ask(round_no=0))
    print(f"  model answer: {rec.get('answer')}")
    print(f"  violations before: {rec.get('violations_before')}  after: {rec.get('violations_after')}"
          f"  -> accepted: {rec['accepted']} {rec.get('reason', '')}")


# ---------------------------------------------------------------- step 5
def step5(question="At what hour of the day does the Night shift start?"):
    from src import graph_rag, vector_rag
    from src.qa import answer_chain, ask, exact_match, lenient_match, load_questions

    known = next((q for q in load_questions() if q["question"].lower() == question.lower()), None)
    title(f"5. Question: {question}" + (f"   (expected answer: {known['answer']})" if known else ""))

    title("5a. Vector RAG: embed the question, take the 4 most similar documents")
    context, docs = vector_rag.retrieve(vector_rag.build_retriever(), question)
    print("  retrieved:", docs)
    out = ask(answer_chain("qa_vector_system"), question, context, "demo_vector")
    print(f"  answer: {out['answer']!r}   evidence: {out['evidence']}")
    if known:
        print(f"  exact={exact_match(out['answer'], known)} lenient={lenient_match(out['answer'], known)}")

    for name, path in [("repaired graph", RESULTS / "phase3_v2_repaired.ttl"), ("ground-truth graph", DATA / "ground_truth.ttl")]:
        title(f"5b. GraphRAG on the {name}: link entities, expand 2 hops, give only fact lines")
        context, info = graph_rag.retrieve(Graph().parse(path), question)
        print(f"  seeds: {info['seeds']}  class seeds: {info['class_seeds']}  fact lines: {info['facts']}")
        print("  first fact lines:\n   " + "\n   ".join(context.splitlines()[1:9]) + "\n   ...")
        out = ask(answer_chain("qa_graph_system"), question, context, "demo_graph")
        print(f"  answer: {out['answer']!r}   evidence: {out['evidence']}")
        if known:
            print(f"  exact={exact_match(out['answer'], known)} lenient={lenient_match(out['answer'], known)}")


if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        print(__doc__)
    else:
        steps[sys.argv[1]](*sys.argv[2:])
