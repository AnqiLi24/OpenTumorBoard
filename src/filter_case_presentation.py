#!/usr/bin/env python3
"""
Read VTT transcripts from $VIDEO_DIR (default data/videos) and use GPT to classify
which videos contain actual patient case presentations.

Output: $FILTER_JSON (default src/case_presentation_filter.json)
Resume-safe: already-classified videos are skipped.

Usage:
    conda run -n mtb python src/filter_case_presentation.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from tqdm import tqdm
from pathlib import Path

from gpt_client import DEPLOYMENT, client

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "data/videos"))
OUTPUT_FILE = Path(os.environ.get("FILTER_JSON", "src/case_presentation_filter.json"))

# First ~12 000 chars covers roughly 10–15 min of speech, enough to tell
# whether specific patient cases are being discussed.
MAX_CHARS = 128_000

_SYSTEM_PROMPT = """\
You are a medical education expert reviewing tumor board video transcripts.
Decide whether the video contains the presentation of one or more specific patient cases.

A video IS a patient case presentation if it discusses:
- A real or de-identified or virtual patient with specific clinical details (age, sex, diagnosis,
  imaging / pathology findings, prior treatments, molecular results, etc.)
- The tumor board is actively deliberating on that individual patient's management.

A video is NOT a case presentation if it is primarily:
- A general educational lecture on drugs, guidelines, or treatment strategies
- A panel debate or Q&A without specific patient data
- Administrative announcements, introductions, or research summaries without
  individual patient deliberation

Respond ONLY with valid JSON in exactly this format (no extra keys):
{
  "is_case_presentation": true,
  "confidence": "high",
  "reason": "one concise sentence"
}"""


def parse_vtt(path: Path) -> str:
    """Strip VTT timestamps and metadata; return plain transcript text."""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*", line):
            continue
        if re.match(r"^\d+$", line):   # numeric cue identifier
            continue
        lines.append(line)
    return " ".join(lines)


def classify(video_name: str, transcript: str) -> dict:
    excerpt = transcript[:MAX_CHARS]
    if len(transcript) > MAX_CHARS:
        excerpt += "\n[transcript truncated]"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Video title: {video_name}\n\nTranscript:\n{excerpt}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main() -> None:
    results: dict = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())
        print(f"Resuming — {len(results)} already classified")

    vtt_files = sorted(VIDEO_DIR.glob("*.vtt"))
    print(f"{len(vtt_files)} VTT files found in {VIDEO_DIR}\n")

    for vtt_path in tqdm(vtt_files):
        stem = vtt_path.stem                          # e.g. "Foo Bar.en"
        video_name = stem[:-3] if stem.endswith(".en") else stem

        if video_name in results:
            label = "YES" if results[video_name].get("is_case_presentation") else "no"
            print(f"  [cached {label}] {video_name}")
            continue

        print(f"  classifying: {video_name} ... ", end="", flush=True)
        transcript = parse_vtt(vtt_path)

        if not transcript.strip():
            results[video_name] = {
                "is_case_presentation": False,
                "confidence": "high",
                "reason": "Empty transcript.",
            }
            print("empty transcript — skipped")
        else:
            try:
                result = classify(video_name, transcript)
                results[video_name] = result
                flag = "YES" if result.get("is_case_presentation") else "no"
                conf = result.get("confidence", "?")
                reason = result.get("reason", "")
                print(f"{flag} ({conf}) — {reason}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                continue   # don't cache errors so they can be retried

        # Write after every video so we can resume safely
        OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_yes = sum(1 for v in results.values() if v.get("is_case_presentation"))
    print(f"\n{'─'*60}")
    print(f"Total classified : {len(results)}")
    print(f"Case presentations: {n_yes}")
    print(f"Results saved to : {OUTPUT_FILE}")

    # print("\nCase presentation videos:")
    # for name, v in sorted(results.items()):
    #     if v.get("is_case_presentation"):
    #         print(f"  [{v.get('confidence', '?'):6s}] {name}")


if __name__ == "__main__":
    main()
