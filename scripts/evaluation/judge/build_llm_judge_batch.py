#!/usr/bin/env python3
"""Build the LLM-judge requests for one or more generation runs.

Board Simulation runs are scored on their extracted conclusion against the
reference conclusion; Specialist Turn runs on their answer against the
specialist's actual answer. The judge sees the case summary, the slide
captions and the slide images alongside the reference and the candidate.

usage: python -m scripts.evaluation.judge.build_llm_judge_batch \
           --task1-run my_model=runs/my_model_board/responses.jsonl \
           --output-dir judge/my_model_board
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.formats import open_jsonl
from scripts.evaluation.formats.qa_evaluation_format import (
    build_multimodal_content,
    extract_answer_text,
)
from scripts.evaluation.formats.simulation_evaluation_format import extract_conclusion

TASK1_RUBRICS = ("conclusion_alignment", "four_decisions")
TASK2_RUBRIC = "clinical_equivalence"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open_jsonl(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=/path/responses.jsonl")
    key, path = value.split("=", 1)
    if not key.strip() or not path.strip():
        raise argparse.ArgumentTypeError("name and path must be non-empty")
    return key.strip(), Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-run", action="append", type=run_argument, default=[],
                        help="Board Simulation run, NAME=path/to/responses.jsonl")
    parser.add_argument("--task2-run", action="append", type=run_argument, default=[],
                        help="Specialist Turn run, NAME=path/to/responses.jsonl")
    parser.add_argument("--task1-rubric", choices=TASK1_RUBRICS,
                        default="conclusion_alignment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, default=Path("data/task1_test.jsonl"))
    parser.add_argument("--task2-manifest", type=Path, default=Path("data/task2_test.jsonl"))
    parser.add_argument("--slide-manifest", type=Path,
                        default=Path("data/test_inputs/board_simulation.image.jsonl"),
                        help="test inputs that list the slide images of every case")
    parser.add_argument("--rubric-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--task1-max-output-tokens", type=int, default=1200)
    parser.add_argument("--task2-max-output-tokens", type=int, default=1024)
    return parser.parse_args()


def task1_user(record: dict[str, Any], candidate: str) -> str:
    return "\n\n".join([
        f"<CASE_SUMMARY>\n{record['case_summary']}\n</CASE_SUMMARY>",
        "<SLIDE_DESCRIPTIONS>\n"
        f"{record.get('slides') or 'No slide descriptions are available.'}\n"
        "</SLIDE_DESCRIPTIONS>",
        f"<REFERENCE_CONCLUSION>\n{record['reference_conclusion']}\n</REFERENCE_CONCLUSION>",
        f"<CANDIDATE_RESPONSE>\n{candidate}\n</CANDIDATE_RESPONSE>",
    ])


def task2_user(record: dict[str, Any], candidate: str) -> str:
    return f"""<TARGET_SPECIALIST_ROLE>
{record['target_specialist_role']}
</TARGET_SPECIALIST_ROLE>

<CASE_SUMMARY>
{record['case_summary']}
</CASE_SUMMARY>

<SLIDE_DESCRIPTIONS>
{record.get('slides') or 'No slide descriptions are available.'}
</SLIDE_DESCRIPTIONS>

<QUESTION>
{record['question']}
</QUESTION>

<REFERENCE_EXPERT_ANSWER>
{record['reference_answer']}
</REFERENCE_EXPERT_ANSWER>

<CANDIDATE_ANSWER>
{candidate}
</CANDIDATE_ANSWER>"""


def load_slide_images(path: Path) -> dict[tuple[str, str], list[str]]:
    images: dict[tuple[str, str], list[str]] = {}
    for row in load_jsonl(path):
        meta = row.get("metadata")
        meta = ast.literal_eval(meta) if isinstance(meta, str) else (meta or {})
        images[(meta["video_uid"], meta["case_id"])] = list(row.get("images") or [])
    missing = [p for paths in images.values() for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} slide images are missing, e.g. {missing[0]}; "
            "see the Benchmark data section of the README"
        )
    return images


def custom_id(task: str, model_key: str, record_id: str) -> str:
    digest = hashlib.sha256(f"{task}\0{model_key}\0{record_id}".encode()).hexdigest()[:20]
    safe_model = "".join(c if c.isalnum() else "_" for c in model_key)[:20]
    return f"{task}_{safe_model}_{digest}"


def request_row(*, row_id: str, prompt: str, user: Any, rubric: str,
                schema: dict[str, Any], max_output_tokens: int,
                args: argparse.Namespace) -> dict[str, Any]:
    return {
        "custom_id": row_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": args.judge_model,
            "input": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": args.reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"opentumorboard_{rubric}",
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
        },
    }


def main() -> None:
    args = parse_args()
    if not args.task1_run and not args.task2_run:
        raise SystemExit("pass at least one --task1-run or --task2-run")
    images = load_slide_images(args.slide_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"judge_model": args.judge_model, "tasks": {}}

    for task, runs, rubric, manifest, id_field in (
        ("task1", args.task1_run, args.task1_rubric, args.task1_manifest, "simulation_id"),
        ("task2", args.task2_run, TASK2_RUBRIC, args.task2_manifest, "qa_id"),
    ):
        if not runs:
            continue
        prompt = (args.rubric_dir / "prompts" / f"{rubric}.txt").read_text(encoding="utf-8").strip()
        schema = json.loads((args.rubric_dir / "schemas" / f"{rubric}.json").read_text(encoding="utf-8"))
        records = load_jsonl(manifest)
        requests: list[dict[str, Any]] = []
        mapping: list[dict[str, Any]] = []
        for model_key, path in runs:
            responses = {str(row[id_field]): row for row in load_jsonl(path)}
            missing = [str(r[id_field]) for r in records if str(r[id_field]) not in responses]
            if missing:
                raise SystemExit(
                    f"{path} has no response for {len(missing)} of {len(records)} "
                    f"test items (e.g. {missing[0]}); rerun generation to fill them"
                )
            for record in records:
                record_id = str(record[id_field])
                response = responses[record_id]
                if task == "task1":
                    candidate, method = extract_conclusion(response)
                else:
                    candidate, method = extract_answer_text(response)
                candidate = candidate.strip() if isinstance(candidate, str) else ""
                entry = {
                    "task": task,
                    "model_key": model_key,
                    "record_id": record_id,
                    "video_uid": record["video_uid"],
                    "case_id": record["case_id"],
                    "qa_type": record.get("qa_type"),
                    "extraction_method": method,
                }
                if not candidate:
                    mapping.append({**entry, "custom_id": None, "candidate_format_failure": True})
                    continue
                user = task1_user(record, candidate) if task == "task1" else task2_user(record, candidate)
                key = (record["video_uid"], record["case_id"])
                if key not in images:
                    raise SystemExit(f"no slide images listed for case {key}")
                user = build_multimodal_content(user, [Path(p) for p in images[key]])
                row_id = custom_id(task, model_key, record_id)
                requests.append(request_row(
                    row_id=row_id, prompt=prompt, user=user, rubric=rubric, schema=schema,
                    max_output_tokens=(args.task1_max_output_tokens if task == "task1"
                                       else args.task2_max_output_tokens),
                    args=args,
                ))
                mapping.append({**entry, "custom_id": row_id, "candidate_format_failure": False})
        write_jsonl(args.output_dir / f"{task}.batch.jsonl", requests)
        write_jsonl(args.output_dir / f"{task}.judge_manifest.jsonl", mapping)
        config["tasks"][task] = {
            "rubric": rubric,
            "runs": {key: str(path) for key, path in runs},
            "requests": len(requests),
            "candidate_format_failures": sum(row["candidate_format_failure"] for row in mapping),
        }

    (args.output_dir / "build_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
