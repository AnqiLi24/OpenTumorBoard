#!/usr/bin/env python3
"""Build the Board Simulation finetuning set (ShareGPT format, slide images attached).

Each training case becomes one conversation:
  system  the Board Simulation prompt (evaluation/prompts/board_simulation_image.txt)
  human   CASE SUMMARY + SLIDES, with one <image> marker per slide
  gpt     the recorded discussion, one numbered turn per utterance in time order,
          followed by <conclusion>reference_conclusion</conclusion>

Every training (or validation) case of the released split is used when its
recorded discussion, its conclusion and all of its slide images are available.
The recorded discussion is not redistributed. It is read from the curation
pipeline's output for the same recording, data/processed/<video_dir>/align_<case_id>.json,
unless the input records already carry a `reference_discussion` field.

usage:
  python training/build_sft_data.py --split train \
      --records data/task1_train.jsonl --processed-root data/processed \
      --image-root data --out training/data/board_sft_train.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

MARKER = "<image>"
REPO = Path(__file__).resolve().parents[1]


def recorded_discussion(record: dict, processed_root: Path | None) -> list[dict]:
    if record.get("reference_discussion"):
        return record["reference_discussion"]
    if processed_root is None:
        raise SystemExit("records carry no reference_discussion; pass --processed-root")
    path = processed_root / record["video_dir"] / f"align_{record['case_id']}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("aligned_utterances") or []


def discussion_block(discussion: list[dict]) -> str:
    lines, n = ["<discussion>"], 0
    for u in sorted(discussion, key=lambda u: float(u.get("start_sec") or 0)):
        role = str(u.get("inferred_role") or "").strip().lower() or "other"
        text = " ".join(str(u.get("text") or "").split())
        if not text:
            continue
        n += 1
        lines.append(f"<turn {n}> {role.title()}: {text}")
    if n == 0:
        return ""
    lines.append("</discussion>")
    return "\n".join(lines)


def open_records(path: Path):
    """Open a .jsonl file, or the .jsonl.gz shipped in its place."""

    if path.exists():
        return path.open(encoding="utf-8")
    packed = path.with_name(path.name + ".gz")
    if packed.exists():
        return gzip.open(packed, "rt", encoding="utf-8")
    raise FileNotFoundError(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("train", "validation"))
    ap.add_argument("--processed-root", type=Path, default=None)
    ap.add_argument("--image-root", type=Path, required=True,
                    help="directory holding slides/<video_dir>/frames/*.jpg")
    ap.add_argument("--slides-index", type=Path, default=REPO / "data/slides_index.jsonl")
    ap.add_argument("--prompt", type=Path,
                    default=REPO / "evaluation/prompts/board_simulation_image.txt")
    ap.add_argument("--prompt-version", default="board_simulation_image")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    images_by_id = {}
    for line in open_records(a.slides_index):
        row = json.loads(line)
        if row["task"] == "task1" and row["split"] == a.split:
            images_by_id[row["id"]] = row["images"]
    system_prompt = a.prompt.read_text(encoding="utf-8").strip()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with a.out.open("w", encoding="utf-8") as out:
        for line in open_records(a.records):
            if not line.strip():
                continue
            record = json.loads(line)
            conclusion = (record.get("reference_conclusion") or "").strip()
            discussion = discussion_block(recorded_discussion(record, a.processed_root))
            slides = str(record.get("slides") or "")
            user_turn = f"CASE SUMMARY:\n{record.get('case_summary') or ''}\n\nSLIDES:\n{slides}"
            listed = images_by_id.get(record["simulation_id"])
            images = [str(a.image_root / p) for p in (listed or [])]
            # A case is used only with its discussion, its conclusion and every slide image.
            if (not conclusion or not discussion or listed is None or not listed
                    or user_turn.count(MARKER) != len(listed)
                    or not all(os.path.isfile(p) for p in images)):
                skipped += 1
                continue
            row = {
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": user_turn},
                    {"from": "gpt", "value": f"{discussion}\n<conclusion>{conclusion}</conclusion>"},
                ],
                "prompt_version": a.prompt_version,
                "condition": "multimodal",
                "set": "task1_discussion_conclusion",
                "source_id": record["simulation_id"],
                "video_uid": record.get("video_uid"),
                "case_id": record.get("case_id"),
                "images": images,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "skipped": skipped, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
