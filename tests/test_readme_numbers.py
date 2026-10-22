"""Phase 5 traceability check: every numeric cell in a README table must exist in a results JSON file.

Only cells that are purely numbers ("0.742", "54", "0.7 / 0.8") are checked; text cells are skipped.
"""
import json
import re

from src.config import RESULTS, ROOT

NUM_CELL = re.compile(r"^-?\d+(\.\d+)?( / -?\d+(\.\d+)?)?$")


def numbers_in(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(round(float(obj), 6))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)


def result_numbers() -> set:
    out = set()
    for path in RESULTS.glob("*.json"):  # top-level result files only
        numbers_in(json.loads(path.read_text(encoding="utf-8")), out)
    return out


def readme_numeric_cells() -> list[str]:
    cells = []
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        for cell in line.strip("|").split("|"):
            cell = cell.strip().strip("*")
            if NUM_CELL.match(cell):
                cells.append(cell)
    return cells


def test_readme_table_numbers_come_from_results():
    known = result_numbers()
    cells = readme_numeric_cells()
    assert len(cells) > 100  # the tables are really being read
    missing = [c for c in cells for x in c.split(" / ") if round(float(x), 6) not in known]
    assert not missing, f"README numbers not found in results/*.json: {missing}"
