"""Phase 4 shared parts: questions, the answer format, the LLM call, and scoring.

Note: the JSON format is also written in the system prompts. Ollama's schema
constraint only restricts decoding; the model itself never sees the schema.

Two scores per answer:
- exact:   the normalised answer equals the expected answer
           (numbers: only a number, optionally with a unit, equal to the expected value;
            names: the code or the full name, e.g. "K1" or "Packer K1")
- lenient: numbers within 2 %; names: the expected code appears somewhere in the answer;
           text: the expected text appears in the answer
"""
import json
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.config import DATA, load_prompt
from src.extract_schema import Strict
from src.llm import get_llm, log_llm_call

UNITS = r"(kw|s|sec|seconds|h|hours|parts|pcs)?"
NUMBER = r"(?<![A-Za-z])-?\d+(?:\.\d+)?"  # a number not glued to letters (so not the 2 in "P2")


class Answer(Strict):
    # Evidence first, answer second: the model first copies the facts it relies on,
    # then answers. Run v1 had the answer first and produced garbled answers
    # (results/phase4_v1_*).
    evidence: list[str] = Field(description="The facts or document names used")
    answer: str = Field(description="Short answer: a number without unit, or a name/code, or 'unknown'")


def load_questions() -> list[dict]:
    return json.loads((DATA / "questions.json").read_text(encoding="utf-8"))


def answer_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([("system", load_prompt(system_prompt)),
                                               ("human", load_prompt("qa_user"))])
    return prompt | get_llm().with_structured_output(Answer, method="json_schema", include_raw=True)


def ask(chain, question: str, context: str, condition: str) -> dict:
    start = time.perf_counter()
    result = chain.invoke({"question": question, "context": context})
    seconds = time.perf_counter() - start
    log_llm_call("phase4", condition, {"question": question, "context": context}, result["raw"].content, seconds)
    parsed = result["parsed"]
    if parsed is None:
        return {"answer": "", "evidence": [], "error": str(result["parsing_error"]), "seconds": round(seconds, 1)}
    return {"answer": parsed.answer, "evidence": parsed.evidence, "seconds": round(seconds, 1)}


# ---------- scoring ----------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip().strip(".")).strip()


def compact(s: str) -> str:
    """For codes: 'INC-204' -> 'inc204'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def exact_match(answer: str, q: dict) -> bool:
    a = norm(answer)
    if q["kind"] == "number":
        m = re.fullmatch(rf"(-?\d+(?:\.\d+)?)\s*{UNITS}", a)
        return bool(m) and float(m.group(1)) == float(q["answer"])
    return any(a == norm(alias) or compact(a) == compact(alias) for alias in q["aliases"])


def lenient_match(answer: str, q: dict) -> bool:
    if q["kind"] == "number":
        nums = re.findall(NUMBER, answer.replace(",", ""))
        return bool(nums) and abs(float(nums[0]) - q["answer"]) <= 0.02 * abs(q["answer"]) + 1e-9
    if q["kind"] == "entity":
        tokens = set(re.sub(r"[^a-z0-9]+", " ", answer.lower().replace("-", "")).split())
        tokens |= {t.split("_")[-1] for t in re.findall(r"[a-z]+_[a-z0-9]+", answer.lower())}
        return all(compact(a) in tokens for a in q["accept"])
    return norm(q["answer"]) in norm(answer)
