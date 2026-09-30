#!/usr/bin/env python3
"""Record, without seeing any candidate, which decisions each reference conclusion takes.

The reward reads this cache, so a candidate cannot add or remove a decision from
its own denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset

from reward import extract_reference_parts


HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "source_data" / "reference_scope.txt").read_text().strip()
DIMENSIONS = ("therapy", "surgery", "next_action", "clinical_trial")
SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": "string", "enum": ["positive", "negative", "absent"]} for name in DIMENSIONS},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": [*DIMENSIONS, "rationale"],
    "additionalProperties": False,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _payload(reference: str, evidence: str) -> dict:
    content = (
        f"<CASE_EVIDENCE>\n{evidence}\n</CASE_EVIDENCE>\n\n"
        f"<REFERENCE_CONCLUSION>\n{reference}\n</REFERENCE_CONCLUSION>"
    )
    return {
        "model": os.environ.get("OTB_JUDGE_MODEL", "judge"),
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "otb_reference_scope", "strict": True, "schema": SCHEMA},
        },
    }


def _call(row: dict, retries: int, timeout: float) -> dict:
    _, reference = extract_reference_parts(row["reward_model"]["ground_truth"])
    extra = row["extra_info"]
    endpoint = os.environ.get("OTB_JUDGE_BASE_URL", "http://127.0.0.1:8094/v1").rstrip("/") + "/chat/completions"
    encoded = json.dumps(_payload(reference, str(extra.get("case_evidence") or ""))).encode()
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(endpoint, data=encoded, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
            result = json.loads(raw["choices"][0]["message"]["content"])
            if any(result[name] not in {"positive", "negative", "absent"} for name in DIMENSIONS):
                raise ValueError("invalid scope label")
            return {
                "source_id": str(extra.get("source_id") or extra.get("input_sha256")),
                "reference_sha256": _sha(reference),
                "scope": {name: result[name] for name in DIMENSIONS},
                "rationale": result["rationale"],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"scope judge failed: {error}") from error


def _load_existing(path: Path) -> dict[tuple[str, str], dict]:
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["source_id"], record["reference_sha256"])] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "outputs/rl/reference_scope.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("parquet", nargs="*", type=Path,
                        default=[HERE / "data/train.parquet", HERE / "data/validation.parquet"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(args.output)
    pending = []
    total = 0
    for path in args.parquet:
        for row in Dataset.from_parquet(str(path)):
            total += 1
            _, reference = extract_reference_parts(row["reward_model"]["ground_truth"])
            extra = row["extra_info"]
            key = (str(extra.get("source_id") or extra.get("input_sha256")), _sha(reference))
            if key not in existing:
                pending.append(row)
    print(f"scope cache: {total - len(pending)}/{total} ready; generating {len(pending)}")
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_call, row, args.retries, args.timeout): row for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"scope generated: {index}/{len(pending)}")
    final = _load_existing(args.output)
    if len(final) < total:
        raise RuntimeError(f"scope cache incomplete: {len(final)} unique records for {total} rows")
    print(f"REFERENCE_SCOPE_V13_OK records={len(final)} output={args.output}")


if __name__ == "__main__":
    main()
