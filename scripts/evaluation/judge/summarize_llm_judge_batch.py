#!/usr/bin/env python3
"""Summarize a judged batch into per-model scores.

Board Simulation reports conclusion alignment (1-5), or with the four-decision
rubric the mean of therapy, surgery, next action and clinical trial (1-5) over
the cases whose reference takes that decision. Specialist Turn reports clinical
equivalence (1-5) with its critical-error and unsupported-claim rates.

A response with no extractable conclusion or answer is scored at 1, the floor
of both 1-5 scales. Judge failures are excluded and counted.

usage: python -m scripts.evaluation.judge.summarize_llm_judge_batch --batch-dir judge/my_model_board
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

FORMAT_FAILURE_SCORE = 1
DECISIONS = ("therapy", "surgery", "next_action", "clinical_trial")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_text(body: dict[str, Any]) -> str:
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("response body has no output_text content")


def score(value: Any, low: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= 5:
        raise ValueError(f"score {value!r} is out of range")
    return value


def validate(rubric: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("judge result must be an object")
    if rubric == "conclusion_alignment":
        if set(result) != {"conclusion_alignment"}:
            raise ValueError("unexpected keys")
        score(result["conclusion_alignment"]["score"])
    elif rubric == "four_decisions":
        if set(result) != {*DECISIONS, "rationale"}:
            raise ValueError("unexpected keys")
        for decision in DECISIONS:
            value = result[decision]
            # -1 marks a decision the reference does not take.
            if value != -1:
                score(value)
    else:
        if set(result) != {"clinical_equivalence_score", "critical_error",
                           "unsupported_claim", "rationale"}:
            raise ValueError("unexpected keys")
        score(result["clinical_equivalence_score"])
        for flag in ("critical_error", "unsupported_claim"):
            if not isinstance(result[flag], bool):
                raise ValueError(f"{flag} must be boolean")
    return result


def judge_rows(batch_dir: Path, task: str, rubric: str) -> list[dict[str, Any]]:
    outputs = {row["custom_id"]: row for row in load_jsonl(batch_dir / f"{task}.output.jsonl")}
    rows = []
    for mapping in load_jsonl(batch_dir / f"{task}.judge_manifest.jsonl"):
        if mapping["candidate_format_failure"]:
            rows.append({**mapping, "status": "format_failure"})
            continue
        output = outputs.get(mapping["custom_id"])
        response = (output or {}).get("response") or {}
        if output is None or output.get("error") or response.get("status_code") != 200:
            rows.append({**mapping, "status": "judge_failure"})
            continue
        try:
            result = validate(rubric, json.loads(output_text(response.get("body") or {})))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            rows.append({**mapping, "status": "judge_failure", "error": str(exc)})
            continue
        rows.append({**mapping, "status": "scored", "result": result})
    return rows


def mean_with_floor(scores: list[int], failures: int) -> float | None:
    total = len(scores) + failures
    return (sum(scores) + FORMAT_FAILURE_SCORE * failures) / total if total else None


def rate(rows: list[dict[str, Any]], flag: str) -> float | None:
    return sum(row["result"][flag] for row in rows) / len(rows) if rows else None


def summarize(rubric: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["status"] == "scored"]
    failures = sum(row["status"] == "format_failure" for row in rows)
    entry: dict[str, Any] = {
        "items": len(rows),
        "scored": len(scored),
        "format_failures": failures,
        "judge_failures": sum(row["status"] == "judge_failure" for row in rows),
    }
    if rubric == "conclusion_alignment":
        entry["conclusion_alignment"] = mean_with_floor(
            [row["result"]["conclusion_alignment"]["score"] for row in scored], failures)
    elif rubric == "four_decisions":
        for decision in DECISIONS:
            values = [row["result"][decision] for row in scored if row["result"][decision] >= 0]
            entry[decision] = {"mean": fmean(values) if values else None, "cases": len(values)}
    else:
        entry["clinical_equivalence"] = mean_with_floor(
            [row["result"]["clinical_equivalence_score"] for row in scored], failures)
        entry["critical_error_rate"] = rate(scored, "critical_error")
        entry["unsupported_claim_rate"] = rate(scored, "unsupported_claim")
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            by_type[str(row["qa_type"])].append(row)
        entry["by_question_type"] = {
            qa_type: {
                "count": len(items),
                "clinical_equivalence": fmean(
                    row["result"]["clinical_equivalence_score"] for row in items),
                "critical_error_rate": rate(items, "critical_error"),
                "unsupported_claim_rate": rate(items, "unsupported_claim"),
            }
            for qa_type, items in sorted(by_type.items())
        }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.batch_dir / "build_config.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    for task, info in config["tasks"].items():
        rubric = info["rubric"]
        rows = judge_rows(args.batch_dir, task, rubric)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[row["model_key"]].append(row)
        summary[task] = {
            "rubric": rubric,
            "models": {model: summarize(rubric, items) for model, items in sorted(by_model.items())},
        }
        with (args.batch_dir / f"{task}.per_item.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.batch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
