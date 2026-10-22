"""Phase 0: smoke test. One prompt -> LangChain -> Ollama -> answer.

Runs the same question three times to check that temperature 0 + fixed seed
really gives identical output. Writes results/phase0_hello.json.
Run:  python -m src.hello
"""
import json

from langchain_core.prompts import ChatPromptTemplate

from src.config import MODEL, RESULTS, SEED, TEMPERATURE, load_prompt
from src.llm import get_llm, run_logged

QUESTION = "What does a buffer do between two machines in a production line?"


def main() -> None:
    # A LangChain "chain" = prompt template piped into a model.
    prompt = ChatPromptTemplate.from_template(load_prompt("hello"))
    chain = prompt | get_llm()

    runs = []
    for i in range(3):
        answer = run_logged(chain, {"question": QUESTION}, phase="phase0", prompt_name="hello")
        runs.append(answer)
        print(f"Run {i + 1}: {answer}")

    result = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "question": QUESTION,
        "answers": runs,
        "identical": len(set(runs)) == 1,
    }
    (RESULTS / "phase0_hello.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Identical across runs:", result["identical"])


if __name__ == "__main__":
    main()
