"""
Go / no-go: check every measured number against the threshold it has to clear.

WHY A SCRIPT AND NOT A PARAGRAPH
    "The system performs well" is not an acceptance criterion. A criterion names the
    measurement, the threshold and what happens if it is not met. This reads the evaluation
    reports the other scripts wrote and prints PASS or FAIL for each one, so the question
    "is this thing finished?" has an answer that is not an opinion.

    A missing report is reported as SKIPPED, not as a pass. Nothing here invents a number.

Run:
    python -m scripts.check_acceptance
    python -m scripts.check_acceptance --json     # for CI or the report appendix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs" / "evaluation"
CRITERIA_FILE = ROOT / "configs" / "acceptance.yaml"

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"

OPERATORS = {
    ">=": lambda value, threshold: value >= threshold,
    ">": lambda value, threshold: value > threshold,
    "<=": lambda value, threshold: value <= threshold,
    "<": lambda value, threshold: value < threshold,
    "==": lambda value, threshold: value == threshold,
}


def dig(data, path: Sequence) -> Optional[object]:
    """Walk a nested report by key path. Returns None if any step is missing."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def check_one(criterion: dict, reports_dir: Path) -> dict:
    result = {"id": criterion.get("id", "?"), "what": criterion.get("what", ""),
              "why": criterion.get("why", ""), "operator": criterion.get("operator", ">="),
              "threshold": criterion.get("threshold"), "report": criterion.get("report", ""),
              "value": None, "status": SKIPPED, "detail": ""}

    path = reports_dir / str(criterion.get("report", ""))
    if not path.exists():
        result["detail"] = "report not produced yet"
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["detail"] = f"report is not valid JSON: {exc}"
        return result

    value = dig(data, criterion.get("path", []))
    if value is None or isinstance(value, (dict, list)):
        result["detail"] = f"no value at {' -> '.join(map(str, criterion.get('path', [])))}"
        return result

    compare = OPERATORS.get(result["operator"])
    if compare is None:
        result["detail"] = f"unknown operator {result['operator']}"
        return result

    result["value"] = value
    result["status"] = PASS if compare(value, criterion["threshold"]) else FAIL
    return result


def load_criteria(path: Path) -> List[dict]:
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return list(data.get("criteria", []))


def verdict(results: List[dict]) -> str:
    if any(r["status"] == FAIL for r in results):
        return "NO-GO"
    if any(r["status"] == SKIPPED for r in results):
        return "INCOMPLETE"
    return "GO"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--criteria", type=Path, default=CRITERIA_FILE)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if anything is SKIPPED as well as FAILED")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = [check_one(criterion, args.reports) for criterion in load_criteria(args.criteria)]
    overall = verdict(results)

    if args.json:
        print(json.dumps({"verdict": overall, "criteria": results}, indent=2))
    else:
        print(f"{'':<5}{'criterion':<26}{'measured':>12}{'':<3}{'required':<14}{'status'}")
        for result in results:
            mark = {PASS: "OK", FAIL: "XX", SKIPPED: "--"}[result["status"]]
            measured = "-" if result["value"] is None else f"{result['value']}"
            required = f"{result['operator']} {result['threshold']}"
            print(f"{mark:<5}{result['id']:<26}{measured:>12}   {required:<14}{result['status']}"
                  + (f"  ({result['detail']})" if result["detail"] else ""))
        print(f"\nVERDICT: {overall}")
        if overall == "INCOMPLETE":
            print("Some measurements have not been produced yet. A missing report is not a pass.")
        if overall == "NO-GO":
            print("At least one claim the project makes is not supported by its own measurement.")

    if overall == "NO-GO":
        return 1
    if overall == "INCOMPLETE" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
