#!/usr/bin/env python3
"""Task 1 generation WITH the slide images. A separate runner, deliberately.

WHY NOT THE SHARED RUNNER. run_vllm_simulation_evaluation.py sends `messages` as two
plain strings and has no notion of an image; and normalize_input_record rebuilds the
user turn from USER_TEMPLATE, which hardcodes the heading "SLIDE DESCRIPTIONS:". The
multimodal prompt's own one-shot example uses "SLIDES:", so the multimodal manifest
cannot satisfy that byte-equality check. Relaxing the check for one experiment would
trade a permanent safety property - the thing that makes a published run provably the
prompt it declares - for a single measurement, so the shared runner and the format
module are left untouched and the 15 caption-only rows are unaffected by this file
existing.

WHAT REPLACES THE INVARIANT. A strict one of its own, asserted at load before any
request is sent:

  * the system turn must equal the --prompt file, stripped, byte for byte
  * the number of <image> markers in the human turn must equal len(images)
  * every image file must exist
  * the text parts, rejoined with the markers put back, must reproduce the human turn
    byte for byte - so the interleaving cannot silently drop or reorder a caption

THE SPLICING. images[n] is the n-th <image> in the human turn, so the user content is
built by splitting on the marker and interleaving: text, image, text, image, ... That
keeps each slide next to its own caption. Sending all images first and the text after
would preserve the count and destroy the pairing, and nothing downstream would notice.

usage.prompt_tokens is recorded per response, which is the check that the images
actually arrived: a text-only Task 1 prompt is about 1,700 tokens, and a request whose
images were dropped lands there while still returning a plausible discussion.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.evaluation.formats import jsonl_path, open_jsonl
from scripts.evaluation.generation.run_vllm_simulation_evaluation import (  # noqa: E402
    extract_final_response, parse_simulation)

MARKER = "<image>"
GENERATOR = "board_simulation_image"
ANTHROPIC_VERSION = "2023-06-01"


def to_anthropic_blocks(parts: list[dict]) -> list[dict]:
    """Convert OpenAI-shaped content parts into Messages API blocks.

    Shape translation only: order, split points and the caption paired with each
    image are unchanged, so the build_content invariant holds on this path too.
    """
    blocks: list[dict] = []
    for part in parts:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
            continue
        head, _, payload = part["image_url"]["url"].partition(",")
        media_type = head[len("data:"):].split(";")[0]
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": payload}})
    return blocks


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_content(human: str, images: list[Path]) -> list[dict]:
    """Interleave text and images so each slide sits beside its own caption."""
    parts = human.split(MARKER)
    if len(parts) - 1 != len(images):
        raise ValueError(f"{len(parts)-1} markers but {len(images)} images")
    rejoined = MARKER.join(parts)
    if rejoined != human:
        raise ValueError("splitting on the marker did not round-trip")
    content: list[dict] = []
    for index, chunk in enumerate(parts):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if index < len(images):
            content.append({"type": "image_url",
                            "image_url": {"url": data_uri(images[index])}})
    return content


def load_manifest(path: Path, prompt: str) -> list[dict[str, Any]]:
    with open_jsonl(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        turns = row["conversations"]
        if [t["from"] for t in turns] != ["system", "human"]:
            raise SystemExit(f"{row['id']}: expected system+human turns")
        if turns[0]["value"].strip() != prompt:
            raise SystemExit(f"{row['id']}: system turn is not the --prompt file")
        images = [Path(p) for p in (row.get("images") or [])]
        missing = [str(p) for p in images if not p.is_file()]
        if missing:
            raise SystemExit(f"{row['id']}: missing image files {missing[:2]}")
        human = turns[1]["value"]
        row["_content"] = build_content(human, images)
        row["_text_messages"] = [{"role": "system", "content": turns[0]["value"]},
                                 {"role": "user", "content": human}]
        row["_images"] = [str(p) for p in images]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/test_inputs/board_simulation.image.jsonl"))
    ap.add_argument("--prompt", type=Path,
                    default=Path("evaluation/prompts/board_simulation_image.txt"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True,
                    help="OpenAI-compatible endpoint serving the model")
    ap.add_argument("--provider",
                    choices=["local_vllm", "openai", "openrouter", "anthropic"],
                    default="local_vllm")
    ap.add_argument("--api-key-file", type=Path,
                    help="file holding the API key; required for hosted providers")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning effort for hosted reasoning models; omitted when not given")
    ap.add_argument("--no-temperature", action="store_true",
                    help="do not send temperature (models that accept only the default)")
    ap.add_argument("--thinking", choices=["adaptive", "disabled"], default=None,
                    help="anthropic only: adaptive thinking")
    ap.add_argument("--model", required=True)
    ap.add_argument("--resolved-model", help="default: --model")
    ap.add_argument("--model-revision", default="unspecified")
    ap.add_argument("--prompt-version", help="default: the prompt file name")
    ap.add_argument("--parameter-profile", default="default")
    ap.add_argument("--response-mode", default="instruction_following",
                    choices=("instruction_following", "reasoning_final"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-ids", nargs="*", default=[],
                    help="probe mode: run just these simulation ids")
    ap.add_argument("--strip-images", action="store_true",
                    help="probe mode: send the SAME records with the image parts "
                         "removed, so the per-record token difference isolates "
                         "what the images cost on THIS model")
    a = ap.parse_args()
    if a.resolved_model is None:
        a.resolved_model = a.model
    if a.prompt_version is None:
        a.prompt_version = a.prompt.stem

    hosted = a.provider in ("openai", "openrouter", "anthropic")
    if hosted and not a.api_key_file:
        raise SystemExit(f"--provider {a.provider} requires --api-key-file")
    headers = {"Content-Type": "application/json"}
    if a.api_key_file:
        token = a.api_key_file.read_text().strip()
        if a.provider == "anthropic":
            headers["x-api-key"] = token
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = "Bearer " + token
    gen_params: dict[str, Any] = {"top_p": a.top_p, "seed": a.seed}
    if not a.no_temperature:
        gen_params["temperature"] = a.temperature
    gen_params["max_completion_tokens" if a.provider == "openai"
               else "max_tokens"] = a.max_tokens
    if a.reasoning_effort is not None:
        if a.provider == "openrouter":
            gen_params["reasoning"] = {"effort": a.reasoning_effort}
        else:
            gen_params["reasoning_effort"] = a.reasoning_effort
    if a.provider == "anthropic":
        gen_params.pop("top_p", None)
        gen_params.pop("seed", None)
        if a.thinking is not None:
            gen_params["thinking"] = {"type": a.thinking}

    prompt = a.prompt.read_text(encoding="utf-8").strip()
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    rows = load_manifest(a.manifest, prompt)
    if a.only_ids:
        rows = [r for r in rows if str(r["id"]) in set(a.only_ids)]
        if len(rows) != len(a.only_ids):
            raise SystemExit(f"asked for {len(a.only_ids)} ids, matched {len(rows)}")
    elif a.limit:
        rows = rows[: a.limit]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = a.output_dir / "responses.jsonl"
    done = set()
    if responses_path.exists():
        done = {json.loads(l)["simulation_id"] for l in
                responses_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if str(r["id"]) not in done]

    (a.output_dir / "run_config.json").write_text(json.dumps({
        "schema_version": 1, "task_name": "tumor_board_simulation",
        "run_id": a.output_dir.name, "started_utc": utc_now(), "generator": GENERATOR,
        "generator_note": (
            "Multimodal Task 1. NOT produced by run_vllm_simulation_evaluation.py: that "
            "runner sends text-only messages and normalize_input_record rebuilds the user "
            "turn from a template hardcoding 'SLIDE DESCRIPTIONS:', which the multimodal "
            "manifest cannot match. This runner asserts its own invariants at load - "
            "system turn equals the prompt file, marker count equals image count, every "
            "image file exists, and the marker split round-trips - and leaves the shared "
            "runner and the caption-only rows untouched."),
        "condition": "multimodal",
        "manifest": str(a.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(jsonl_path(a.manifest).read_bytes()).hexdigest(),
        "manifest_records_selected": len(rows), "manifest_format": "sharegpt_multimodal",
        "model": a.model, "resolved_model": a.resolved_model,
        "model_revision": a.model_revision, "provider": a.provider,
        "prompt_adapter": "openai_chat_multimodal", "base_url": a.base_url,
        "prompt": str(a.prompt.resolve()), "prompt_sha256": prompt_sha,
        "prompt_version": a.prompt_version,
        "parameter_profile": a.parameter_profile, "response_mode": a.response_mode,
        "image_processing": ("native resolution, no downsampling, no max_pixels and no "
                             "longest-edge limit; each family's own processor rules apply "
                             "and the resulting token counts are recorded per response"),
        "generation": dict(gen_params),
        "concurrency": a.concurrency, "timeout_seconds": a.timeout,
        "max_retries": a.max_retries, "resumed_records": len(done),
    }, indent=2) + "\n", encoding="utf-8")

    endpoint = a.base_url.rstrip("/") + (
        "/messages" if a.provider == "anthropic" else "/chat/completions")
    lock = threading.Lock()
    handle = responses_path.open("a", encoding="utf-8")
    errors_handle = (a.output_dir / "errors.jsonl").open("a", encoding="utf-8")
    counts = {"done": 0, "errors": 0}

    def evaluate(row: dict[str, Any]) -> None:
        content = ([p for p in row["_content"] if p["type"] == "text"]
                   if a.strip_images else row["_content"])
        if a.provider == "anthropic":
            payload = {"model": a.model, **gen_params,
                       "system": row["conversations"][0]["value"],
                       "messages": [{"role": "user",
                                     "content": to_anthropic_blocks(content)}]}
        else:
            payload = {"model": a.model, **gen_params,
                       "messages": [{"role": "system", "content": row["conversations"][0]["value"]},
                                    {"role": "user", "content": content}]}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(1, a.max_retries + 1):
            try:
                request = urllib.request.Request(
                    endpoint, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=a.timeout) as response:
                    body = json.load(response)
                if a.provider == "anthropic":
                    blocks = body.get("content") or []
                    content = "".join(b.get("text", "") for b in blocks
                                      if b.get("type") == "text")
                    thought = "".join(b.get("thinking", "") for b in blocks
                                      if b.get("type") == "thinking")
                    reasoning = thought or None
                    choice = {"finish_reason": body.get("stop_reason")}
                else:
                    choice = body["choices"][0]
                    if choice.get("finish_reason") == "error":
                        raise RuntimeError(
                            "upstream finish_reason=error: "
                            + json.dumps(body.get("error") or {})[:200])
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    reasoning = (message.get("reasoning")
                                 if message.get("reasoning") is not None
                                 else message.get("reasoning_content"))
                extracted, extraction = None, "not_requested"
                if a.response_mode == "reasoning_final":
                    extracted, extraction = extract_final_response(content, reasoning)
                    target = extracted if extracted is not None else ""
                else:
                    target = content
                parsed, status, warnings = parse_simulation(target)
                record = {
                    "kind": "response", "schema_version": 1, "attempt": attempt,
                    "simulation_id": str(row["id"]), "video_uid": row.get("video_uid"),
                    "case_id": row.get("case_id"),
                    "input_format": "sharegpt_multimodal",
                    "images": [] if a.strip_images else row["_images"],
                    "image_count": 0 if a.strip_images else len(row["_images"]),
                    "probe_images_stripped": bool(a.strip_images),
                    "manifest_image_count": len(row["_images"]),
                    "requested_model": a.model, "resolved_model": a.resolved_model,
                    "model_revision": a.model_revision, "provider": a.provider,
                    "prompt_adapter": "openai_chat_multimodal",
                    "prompt_version": a.prompt_version,
                    "parameter_profile": a.parameter_profile,
                    "response_mode": a.response_mode,
                    "raw_response": content, "raw_reasoning_content": reasoning,
                    "extracted_final_response": extracted,
                    "final_response_extraction": extraction,
                    "parsed_output": parsed, "parse_status": status,
                    "parse_warnings": warnings,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage"), "response_id": body.get("id"),
                    "response_model": body.get("model"),
                    "started_utc": utc_now(), "generator": GENERATOR,
                }
                with lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    counts["done"] += 1
                    if counts["done"] % 20 == 0:
                        print(json.dumps({"completed": counts["done"],
                                          "remaining": len(todo) - counts["done"]}), flush=True)
                return
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                    json.JSONDecodeError, TimeoutError, RuntimeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
        with lock:
            errors_handle.write(json.dumps({"simulation_id": str(row["id"]),
                                            "error": last}, ensure_ascii=False) + "\n")
            errors_handle.flush()
            counts["errors"] += 1

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        list(pool.map(evaluate, todo))
    handle.close(); errors_handle.close()

    total = len(done) + counts["done"]
    (a.output_dir / "summary.json").write_text(json.dumps({
        "schema_version": 1, "task_name": "tumor_board_simulation",
        "run_id": a.output_dir.name, "updated_utc": utc_now(),
        "final": counts["errors"] == 0 and total == len(rows),
        "selected": len(rows), "completed_total": total, "errors": counts["errors"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(rows), "completed": total,
                      "errors": counts["errors"]}, indent=1))
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
