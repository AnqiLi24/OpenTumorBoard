#!/usr/bin/env python3
"""Run a resumable concurrent QA evaluation against an OpenAI-compatible API."""

from __future__ import annotations


import sys
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import random
import re
import signal
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.evaluation.formats import jsonl_path, open_jsonl
from scripts.evaluation.formats.deepseek_v4_format import encode_deepseek_v4_chat, encode_deepseek_v4_thinking
from scripts.evaluation.formats.qa_evaluation_format import (
    manifest_format, normalize_input_record, normalize_multimodal_record)
from scripts.evaluation.formats.reasoning_output_format import extract_final_response


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_content(
    content: str, output_schema: str
) -> tuple[dict[str, Any] | None, str, str | None]:
    cleaned = content.strip()
    without_thinking = re.sub(
        r"\A\s*<think\b[^>]*>.*?</think>\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    removed_thinking = without_thinking != cleaned
    cleaned = without_thinking
    candidates = [cleaned]
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())
    left, right = cleaned.find("{"), cleaned.rfind("}")
    if left >= 0 and right > left:
        candidates.append(cleaned[left : right + 1])
    last_error: Exception | None = None
    for index, candidate in enumerate(dict.fromkeys(candidates)):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise TypeError("top-level output is not an object")
            expected_keys = (
                {"answer"}
                if output_schema == "answer_only"
                else {"answer", "insufficient_evidence"}
            )
            if set(parsed) != expected_keys:
                raise ValueError("output keys do not exactly match the required schema")
            if not isinstance(parsed["answer"], str):
                raise TypeError("answer is not a string")
            if (
                output_schema == "answer_with_insufficient"
                and not isinstance(parsed["insufficient_evidence"], bool)
            ):
                raise TypeError("insufficient_evidence is not a boolean")
            status = (
                "parsed"
                if index == 0 and not removed_thinking
                else "parsed_after_cleanup"
            )
            return parsed, status, None
        except Exception as exc:  # Preserve invalid model output without rerunning it.
            last_error = exc
    assert last_error is not None
    return None, "raw_fallback", f"{type(last_error).__name__}: {last_error}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qa_id = record.get("qa_id")
            if qa_id:
                completed.add(str(qa_id))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/test_inputs/specialist_turn.image.jsonl"))
    parser.add_argument("--prompt", type=Path,
                        default=Path("evaluation/prompts/specialist_turn_image.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible endpoint serving the model")
    parser.add_argument("--provider", default="local_vllm")
    parser.add_argument(
        "--prompt-adapter",
        choices=("openai_chat", "deepseek_v4_chat", "deepseek_v4_thinking"),
        default="openai_chat",
        help=(
            "Use OpenAI chat completions, or an official DeepSeek-V4 chat or "
            "thinking encoding with the completions endpoint."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment-variable name containing a Bearer API key; the value is never stored.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="File containing a Bearer API key; the secret value is never stored in run artifacts.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--resolved-model", help="default: --model")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--prompt-version", help="default: the prompt file name")
    parser.add_argument("--parameter-profile", default="default")
    parser.add_argument(
        "--response-mode",
        choices=("instruction_following", "reasoning_final"),
        default="instruction_following",
        help=(
            "Parse the complete response, or preserve visible reasoning while "
            "parsing only the suffix after its final-answer heading."
        ),
    )
    parser.add_argument(
        "--output-schema",
        choices=("answer_only", "answer_with_insufficient"),
        default="answer_only",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--omit-seed",
        action="store_true",
        help="Do not send seed to providers whose API does not support it.",
    )
    parser.add_argument(
        "--thinking",
        choices=("omit", "enabled", "disabled"),
        default="omit",
        help="Optional OpenAI-compatible thinking toggle used by DeepSeek V4.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.resolved_model is None:
        args.resolved_model = args.model
    if args.prompt_version is None:
        args.prompt_version = args.prompt.stem
    return args


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.api_key_env and args.api_key_file:
        raise SystemExit("Use only one of --api-key-env or --api-key-file")
    if args.prompt_adapter.startswith("deepseek_v4_") and args.thinking != "omit":
        raise SystemExit(
            "DeepSeek-V4 prompt adapters already fix the reasoning mode; omit --thinking"
        )
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"Required API key environment variable is not set: {args.api_key_env}")
    if args.api_key_file:
        try:
            api_key = args.api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Could not read API key file: {exc}") from exc
        if not api_key:
            raise SystemExit("API key file is empty")
    manifest_bytes = jsonl_path(args.manifest).read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    prompt_sha256 = sha256_bytes(system_prompt.encode("utf-8"))
    raw_manifest = load_jsonl(args.manifest)
    input_format = manifest_format(raw_manifest)
    # A record carrying an `images` KEY is the multimodal condition, empty list
    # included. Detected rather than flagged, because the manifest is the thing
    # that knows: a flag can disagree with the file it is pointed at, and the two
    # conditions must never be silently swapped.
    #
    # Key presence, not truthiness. Four cases in the test split have no slides at
    # all, so 83 of the 4,844 questions carry `images: []`, and a truthiness test
    # reads those as caption-only records and rejects the whole manifest as mixed.
    # Empty-list and key-absent are different facts: one says this case has no
    # slides, the other says this file is not the multimodal condition. Those 83
    # are the placebo group - their two arms differ only by the system prompt and
    # the SLIDES heading, verified byte for byte - so dropping or rejecting them
    # would throw away the one control that separates a prompt effect from an
    # image effect.
    multimodal = any("images" in record for record in raw_manifest)
    if multimodal:
        if not all("images" in record for record in raw_manifest):
            raise SystemExit("manifest mixes multimodal and caption-only records")
        input_format = "sharegpt_multimodal"
        manifest = [
            normalize_multimodal_record(record, system_prompt)
            for record in raw_manifest
        ]
    else:
        manifest = [
            normalize_input_record(record, system_prompt) for record in raw_manifest
        ]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    existing = completed_ids(responses_path)
    pending = [record for record in manifest if record["qa_id"] not in existing]
    started_utc = utc_now()
    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": None if args.omit_seed else args.seed,
    }
    if args.thinking != "omit":
        generation["thinking"] = args.thinking
    run_config = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "started_utc": started_utc,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "manifest_records_selected": len(manifest),
        "manifest_format": input_format,
        "resumed_records": len(existing),
        "model": args.model,
        "resolved_model": args.resolved_model,
        "model_revision": args.model_revision,
        "provider": args.provider,
        "base_url": args.base_url,
        "prompt_adapter": args.prompt_adapter,
        "api_key_env": args.api_key_env,
        "api_key_file": str(args.api_key_file.resolve()) if args.api_key_file else None,
        "prompt": str(args.prompt.resolve()),
        "prompt_version": args.prompt_version,
        "prompt_sha256": prompt_sha256,
        "parameter_profile": args.parameter_profile,
        "response_mode": args.response_mode,
        "output_schema": args.output_schema,
        "generation": generation,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "manifest_sha256",
            "manifest_format",
            "model",
            "model_revision",
            "prompt_sha256",
            "parameter_profile",
            "response_mode",
            "output_schema",
            "generation",
            "provider",
            "base_url",
            "prompt_adapter",
        )
        for key in invariant_keys:
            previous_value = previous.get(key)
            if key == "manifest_format" and previous_value is None:
                previous_value = "custom"
            if key == "prompt_adapter" and previous_value is None:
                previous_value = "openai_chat"
            if previous_value != run_config.get(key):
                raise SystemExit(f"Refusing incompatible resume: {key} changed")
        run_config["initial_started_utc"] = previous.get(
            "initial_started_utc", previous.get("started_utc")
        )
    atomic_json(config_path, run_config)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    endpoint_suffix = (
        "/completions"
        if args.prompt_adapter.startswith("deepseek_v4_")
        else "/chat/completions"
    )
    endpoint = args.base_url.rstrip("/") + endpoint_suffix

    def evaluate(record: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.prompt_adapter == "deepseek_v4_chat":
            payload["prompt"] = encode_deepseek_v4_chat(record["_messages"])
        elif args.prompt_adapter == "deepseek_v4_thinking":
            payload["prompt"] = encode_deepseek_v4_thinking(record["_messages"])
        else:
            payload["messages"] = record["_messages"]
        if not args.omit_seed:
            payload["seed"] = args.seed
        if args.thinking != "omit":
            payload["thinking"] = {"type": args.thinking}
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: str | None = None
        for attempt in range(1, args.max_retries + 1):
            if stop.is_set():
                return {"kind": "stopped", "qa_id": record["qa_id"]}
            started_perf = time.perf_counter()
            started_request_utc = utc_now()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(
                endpoint,
                data=request_data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    response_payload = json.load(response)
                elapsed = time.perf_counter() - started_perf
                choice = response_payload["choices"][0]
                message = choice.get("message") or {}
                content = (
                    choice["text"]
                    if args.prompt_adapter.startswith("deepseek_v4_")
                    else (message.get("content") or "")
                )
                reasoning_content = (
                    None
                    if args.prompt_adapter.startswith("deepseek_v4_")
                    else (
                        message.get("reasoning")
                        if message.get("reasoning") is not None
                        else message.get("reasoning_content")
                    )
                )
                extracted_final_response: str | None = None
                final_response_extraction = "not_requested"
                if args.response_mode == "reasoning_final":
                    extracted_final_response, final_response_extraction = (
                        extract_final_response(content, reasoning_content)
                    )
                    if extracted_final_response is None:
                        parsed = None
                        parse_status = "raw_fallback"
                        parse_error = final_response_extraction
                    else:
                        parsed, parse_status, parse_error = parse_content(
                            extracted_final_response, args.output_schema
                        )
                        if parse_status == "parsed":
                            parse_status = "parsed_after_cleanup"
                else:
                    parsed, parse_status, parse_error = parse_content(
                        content, args.output_schema
                    )
                run_key_payload = "\x1f".join(
                    (
                        str(record["benchmark_version"]),
                        str(record["qa_id"]),
                        args.resolved_model,
                        args.prompt_version,
                        args.parameter_profile,
                    )
                )
                return {
                    "kind": "response",
                    "schema_version": 1,
                    "run_key": sha256_bytes(run_key_payload.encode("utf-8")),
                    "benchmark_version": record["benchmark_version"],
                    "qa_id": record["qa_id"],
                    "video_uid": record["video_uid"],
                    "case_id": record["case_id"],
                    "qa_type": record["qa_type"],
                    "target_specialist_role": record["target_specialist_role"],
                    "input_format": input_format,
                    "images": record.get("_images", []),
                    "image_count": len(record.get("_images", [])),
                    "provider": args.provider,
                    "prompt_adapter": args.prompt_adapter,
                    "requested_model": args.model,
                    "resolved_model": args.resolved_model,
                    "model_revision": args.model_revision,
                    "prompt_version": args.prompt_version,
                    "parameter_profile": args.parameter_profile,
                    "response_mode": args.response_mode,
                    "output_schema": args.output_schema,
                    "attempt": attempt,
                    "started_utc": started_request_utc,
                    "latency_seconds": elapsed,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response_payload.get("usage"),
                    "raw_response": content,
                    "raw_reasoning_content": reasoning_content,
                    "extracted_final_response": extracted_final_response,
                    "final_response_extraction": final_response_extraction,
                    "parsed_output": parsed,
                    "parse_status": parse_status,
                    "parse_error": parse_error,
                    "response_id": response_payload.get("id"),
                    "response_model": response_payload.get("model"),
                    "system_fingerprint": response_payload.get("system_fingerprint"),
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:4000]
                last_error = f"HTTPError {exc.code}: {body}"
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if not retryable:
                    break
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            if attempt < args.max_retries:
                time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
        return {
            "kind": "error",
            "schema_version": 1,
            "qa_id": record["qa_id"],
            "failed_utc": utc_now(),
            "attempts": args.max_retries,
            "error": last_error,
        }

    counters = {
        "selected": len(manifest),
        "resumed": len(existing),
        "pending_at_start": len(pending),
        "completed_this_process": 0,
        "parsed": 0,
        "parsed_after_cleanup": 0,
        "raw_fallback": 0,
        "errors": 0,
    }
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    process_started = time.monotonic()
    last_report = 0.0

    def write_summary(final: bool = False) -> None:
        elapsed = time.monotonic() - process_started
        done_total = counters["resumed"] + counters["completed_this_process"]
        rate = counters["completed_this_process"] / elapsed if elapsed else 0.0
        remaining = max(0, counters["selected"] - done_total)
        value = {
            "schema_version": 1,
            "run_id": output_dir.name,
            "updated_utc": utc_now(),
            "final": final,
            **counters,
            "completed_total": done_total,
            "remaining": remaining,
            "elapsed_seconds_this_process": elapsed,
            "responses_per_second_this_process": rate,
            "estimated_remaining_seconds": remaining / rate if rate else None,
            "usage_this_process": usage,
            "stop_requested": stop.is_set(),
        }
        atomic_json(summary_path, value)

    if not pending:
        write_summary(final=True)
        print("No pending records; run already complete.", flush=True)
        return

    response_handle = responses_path.open("a", encoding="utf-8", buffering=1)
    error_handle = errors_path.open("a", encoding="utf-8", buffering=1)
    pending_iter = iter(pending)
    futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    window = max(args.concurrency, args.concurrency * 2)

    def refill(executor: concurrent.futures.ThreadPoolExecutor) -> None:
        while not stop.is_set() and len(futures) < window:
            try:
                record = next(pending_iter)
            except StopIteration:
                break
            future = executor.submit(evaluate, record)
            futures[future] = str(record["qa_id"])

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency, thread_name_prefix="qa"
        ) as executor:
            refill(executor)
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, timeout=2.0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    futures.pop(future, None)
                    result = future.result()
                    if result["kind"] == "response":
                        response_handle.write(
                            json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        counters["completed_this_process"] += 1
                        counters[result["parse_status"]] += 1
                        result_usage = result.get("usage") or {}
                        for key in usage:
                            usage[key] += int(result_usage.get(key) or 0)
                    elif result["kind"] == "error":
                        error_handle.write(
                            json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        counters["errors"] += 1
                refill(executor)
                now = time.monotonic()
                if done and (now - last_report >= 10.0):
                    last_report = now
                    write_summary()
                    done_total = counters["resumed"] + counters["completed_this_process"]
                    rate = counters["completed_this_process"] / max(
                        0.001, now - process_started
                    )
                    print(
                        f"completed={done_total}/{counters['selected']} "
                        f"errors={counters['errors']} rate={rate:.2f} QA/s",
                        flush=True,
                    )
                if stop.is_set():
                    for future in futures:
                        future.cancel()
                    break
    finally:
        response_handle.close()
        error_handle.close()
        final = (
            not stop.is_set()
            and counters["resumed"] + counters["completed_this_process"]
            == counters["selected"]
        )
        write_summary(final=final)
    print(json.dumps(json.loads(summary_path.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
