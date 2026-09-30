#!/usr/bin/env python3
"""Convert frozen OpenTumorBoard ShareGPT JSONL files to verl RLHF parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import Dataset


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
OUTPUT = HERE / "data"

TRAIN_FILE = SOURCE / "board_sft_train.jsonl"
VALIDATION_FILE = SOURCE / "board_sft_validation.jsonl"
TEST_FILE = SOURCE / "board_simulation_test.jsonl"
GENERATION_PROMPT_FILE = SOURCE / "board_simulation.txt"

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}
CONCLUSION_RE = re.compile(r"<conclusion>(?P<text>[^<>]+)</conclusion>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_conclusion(response: str) -> str:
    matches = CONCLUSION_RE.findall(response)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one conclusion block, found {len(matches)}")
    return matches[0]


def source_identity(row: dict) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    source_id = str(row.get("source_id") or row.get("id") or metadata.get("simulation_id") or "")
    if not source_id:
        raise ValueError("record is missing its source_id or simulation_id")
    return {
        "source_id": source_id,
        "video_uid": str(row.get("video_uid") or metadata.get("video_uid") or ""),
        "case_id": str(row.get("case_id") or metadata.get("case_id") or ""),
    }


def split_identities(rows: list[dict]) -> set[tuple[str, str]]:
    """Check both record identity and duplicate inputs without exporting hashes."""
    identities = set()
    for row in rows:
        identities.add(("source_id", row["extra_info"]["source_id"]))
        payload = json.dumps(
            {"messages": row["prompt"], "images": row["images"]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        identities.add(("input", hashlib.sha256(payload).hexdigest()))
    return identities


def convert(path: Path, split: str, require_reference: bool, generation_prompt: str) -> list[dict]:
    converted = []
    for index, row in enumerate(read_jsonl(path)):
        conversations = row["conversations"]
        prompt = [
            {"role": ROLE_MAP[turn["from"]], "content": turn["value"]}
            for turn in conversations
            if turn["from"] in {"system", "human"}
        ]
        if [message["role"] for message in prompt] != ["system", "user"]:
            raise ValueError(f"{split}[{index}] does not contain exactly system + user prompt turns")
        # The frozen JSONL remains untouched. Route every local split through one
        # auditable canonical prompt so reward-structure fixes do not require
        # rewriting the source data or thousands of copied JSON strings.
        prompt[0]["content"] = generation_prompt

        assistant = [turn["value"] for turn in conversations if turn["from"] == "gpt"]
        if require_reference and len(assistant) != 1:
            raise ValueError(f"{split}[{index}] expected one reference response, found {len(assistant)}")
        if not require_reference and assistant:
            raise ValueError(f"{split}[{index}] unexpectedly contains a test reference")

        images = list(row.get("images") or [])
        missing = [image for image in images if not Path(image).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}[{index}] has missing image: {missing[0]}")

        system_markers = prompt[0]["content"].count("<image>")
        user_markers = prompt[1]["content"].count("<image>")
        if system_markers != 3:
            raise ValueError(f"{split}[{index}] expected 3 literal system <image> markers, got {system_markers}")
        if user_markers != len(images):
            raise ValueError(
                f"{split}[{index}] has {user_markers} user <image> markers but {len(images)} image paths"
            )

        identity = source_identity(row)
        if assistant:
            # Validate the required block before preserving the complete source
            # transcript. Training rewards both discussion and conclusion; storing
            # only the extracted conclusion here silently made discussion unlearnable.
            extract_conclusion(assistant[0])
        converted.append(
            {
                "data_source": "otb_task1_grpo",
                "prompt": prompt,
                "images": images,
                "ability": "tumor_board",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": assistant[0] if assistant else "",
                },
                "extra_info": {
                    "index": index,
                    "split": split,
                    "n_images": len(images),
                    # Textual case evidence lets the training judge check factual
                    # grounding; evaluation attaches the slide images themselves.
                    "case_evidence": prompt[1]["content"],
                    **identity,
                },
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--generation-prompt",
        type=Path,
        default=GENERATION_PROMPT_FILE,
        help="System prompt to place in every prepared row.",
    )
    args = parser.parse_args()
    missing = [
        path.name
        for path in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} in {SOURCE}. Copy the finetuning data built by "
            "training/build_sft_data.py there, and the released board_simulation.image.jsonl "
            "as board_simulation_test.jsonl."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation_prompt = args.generation_prompt.read_text().strip()

    specs = (
        ("train", TRAIN_FILE, True, 362),
        ("validation", VALIDATION_FILE, True, 56),
        ("test", TEST_FILE, False, 184),
    )
    manifest = {
        "format": "verl_otb_board_simulation_grpo",
        "ground_truth_scope": "complete_discussion_and_conclusion",
        "sources": {},
        "outputs": {},
        "generation_prompt": {
            "path": str(args.generation_prompt),
            "sha256": sha256(args.generation_prompt),
        },
    }
    identity_sets: dict[str, set[tuple[str, str]]] = {}

    for split, source, require_reference, expected_count in specs:
        rows = convert(source, split, require_reference, generation_prompt)
        if len(rows) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} rows, got {len(rows)}")
        output = args.output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(output)
        manifest["sources"][split] = {"path": str(source), "sha256": sha256(source), "rows": len(rows)}
        manifest["outputs"][split] = {"path": str(output), "sha256": sha256(output), "rows": len(rows)}
        identity_sets[split] = split_identities(rows)
        print(f"{split}: {len(rows)} rows -> {output}")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = identity_sets[left] & identity_sets[right]
        if overlap:
            raise ValueError(f"source identity or input leakage between {left} and {right}: {len(overlap)}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
