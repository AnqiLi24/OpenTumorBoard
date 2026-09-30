#!/usr/bin/env python3
"""ROUGE-L and BERTScore of a generation run against the references.

Board Simulation compares the extracted conclusion with the board's conclusion;
Specialist Turn compares the answer with the specialist's answer. ROUGE-L is
the sentence-level longest-common-subsequence F1 over lowercase alphanumeric
tokens, without stemming. BERTScore is the F1 of microsoft/deberta-xlarge-mnli
(layer 40, no idf weighting, no baseline rescaling). A Board Simulation
response without a conclusion counts as 0 in ROUGE-L; every other mean is over
the responses that have text.

usage: python -m scripts.evaluation.metrics.text_metrics --task board \
           --run runs/my_model_board/responses.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

from scripts.evaluation.formats import open_jsonl
from scripts.evaluation.formats.qa_evaluation_format import extract_answer_text
from scripts.evaluation.formats.simulation_evaluation_format import extract_conclusion

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SETTINGS = {
    "board": ("data/task1_test.jsonl", "simulation_id", "reference_conclusion"),
    "turn": ("data/task2_test.jsonl", "qa_id", "reference_answer"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open_jsonl(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens, reference_tokens = tokenize(candidate), tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = lcs_length(candidate_tokens, reference_tokens)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(candidate_tokens), overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def bertscore_f1(candidates: list[str], references: list[str], args: argparse.Namespace) -> list[float]:
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli", num_layers=40, idf=False,
        rescale_with_baseline=False, lang="en", batch_size=args.batch_size,
        device=args.device,
    )
    # Some Transformers releases report a sentinel max length for DeBERTa that
    # overflows the tokenizer; the encoder supports 512 positions.
    scorer._tokenizer.model_max_length = 512
    _, _, f1 = scorer.score(candidates, references, batch_size=args.batch_size, verbose=True)
    return f1.cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=SETTINGS, required=True)
    parser.add_argument("--run", type=Path, required=True, help="responses.jsonl of a generation run")
    parser.add_argument("--manifest", type=Path, help="default: the test file of --task")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest_path, id_field, reference_field = SETTINGS[args.task]
    records = load_jsonl(args.manifest or Path(manifest_path))
    responses = {str(row[id_field]): row for row in load_jsonl(args.run)}
    missing = [str(r[id_field]) for r in records if str(r[id_field]) not in responses]
    if missing:
        raise SystemExit(f"{args.run} has no response for {len(missing)} test items")

    rouge, candidates, references, failures = [], [], [], 0
    for record in records:
        response = responses[str(record[id_field])]
        if args.task == "board":
            candidate, _ = extract_conclusion(response)
        else:
            candidate, _ = extract_answer_text(response)
        candidate = candidate.strip() if isinstance(candidate, str) else ""
        reference = str(record[reference_field]).strip()
        if not candidate:
            failures += 1
            if args.task == "board":
                rouge.append(0.0)
            continue
        rouge.append(rouge_l_f1(candidate, reference))
        candidates.append(candidate)
        references.append(reference)

    summary: dict[str, Any] = {
        "items": len(records),
        "format_failures": failures,
        "rouge_l_f1": fmean(rouge) if rouge else None,
    }
    if not args.skip_bertscore and candidates:
        summary["bertscore_f1"] = fmean(bertscore_f1(candidates, references, args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
