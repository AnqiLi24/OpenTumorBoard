#!/usr/bin/env python3
"""Build auditable verl multimodal-SFT parquet files for OTB Task 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import Dataset


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
DEFAULT_OUTPUT = HERE / "sft_data"
DEFAULT_PROMPT = SOURCE / "board_simulation.txt"
SPLITS = {
    "train": (SOURCE / "board_sft_train.jsonl", 362),
    "validation": (SOURCE / "board_sft_validation.jsonl", 56),
}
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}
DISCUSSION_RE = re.compile(r"<discussion>.*?</discussion>", re.DOTALL)
CONCLUSION_RE = re.compile(r"<conclusion>.*?</conclusion>", re.DOTALL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_identities(rows: list[dict]) -> set[tuple[str, str]]:
    """Check both record identity and duplicate inputs without exporting hashes."""
    identities = set()
    for row in rows:
        identities.add(("source_id", row["source_id"]))
        payload = json.dumps(
            {"messages": row["messages"][:-1], "images": row["images"]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        identities.add(("input", hashlib.sha256(payload).hexdigest()))
    return identities


def convert_split(source: Path, prompt: str, expected_rows: int) -> list[dict]:
    rows = []
    with source.open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            source_row = json.loads(line)
            conversations = source_row["conversations"]
            roles = [turn["from"] for turn in conversations]
            if roles != ["system", "human", "gpt"]:
                raise ValueError(f"{source.name}[{index}] expected system/human/gpt, got {roles}")

            response = conversations[2]["value"]
            if len(DISCUSSION_RE.findall(response)) != 1 or len(CONCLUSION_RE.findall(response)) != 1:
                raise ValueError(f"{source.name}[{index}] does not contain one discussion and conclusion block")

            messages = [
                {"role": ROLE_MAP[turn["from"]], "content": turn["value"]}
                for turn in conversations
            ]
            messages[0]["content"] = prompt
            images = list(source_row.get("images") or [])
            missing = [image for image in images if not Path(image).is_file()]
            if missing:
                raise FileNotFoundError(f"{source.name}[{index}] missing image: {missing[0]}")
            if messages[1]["content"].count("<image>") != len(images):
                raise ValueError(
                    f"{source.name}[{index}] has {messages[1]['content'].count('<image>')} placeholders "
                    f"for {len(images)} images"
                )

            metadata = source_row.get("metadata") or {}
            source_id = str(
                source_row.get("source_id")
                or source_row.get("id")
                or metadata.get("simulation_id")
                or ""
            )
            if not source_id:
                raise ValueError(f"{source.name}[{index}] is missing its source_id or simulation_id")
            rows.append(
                {
                    "messages": messages,
                    "images": images,
                    "source_id": source_id,
                }
            )

    if len(rows) != expected_rows:
        raise ValueError(f"{source}: expected {expected_rows} rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-prompt", type=Path, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    prompt = args.generation_prompt.read_text().strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "verl_otb_board_simulation_sft",
        "assistant_loss_scope": "complete_discussion_and_conclusion",
        "generation_prompt": {
            "path": str(args.generation_prompt),
            "sha256": sha256(args.generation_prompt),
        },
        "splits": {},
    }
    identities = {}

    for split, (source, expected_rows) in SPLITS.items():
        rows = convert_split(source, prompt, expected_rows)
        output = args.output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(output)
        manifest["splits"][split] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "output": str(output),
            "output_sha256": sha256(output),
            "rows": len(rows),
        }
        identities[split] = split_identities(rows)
        print(f"{split}: {len(rows)} -> {output}")

    overlap = identities["train"] & identities["validation"]
    if overlap:
        raise ValueError(f"train/validation leakage by source identity or input: {len(overlap)}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
