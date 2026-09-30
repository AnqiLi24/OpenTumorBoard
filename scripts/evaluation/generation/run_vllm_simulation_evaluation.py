#!/usr/bin/env python3
"""Run resumable Task 1 simulation against an OpenAI-compatible API."""

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
import signal
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.evaluation.formats import jsonl_path, open_jsonl
from scripts.evaluation.formats.deepseek_v4_format import encode_deepseek_v4_chat, encode_deepseek_v4_thinking
from scripts.evaluation.formats.reasoning_output_format import extract_final_response
from scripts.evaluation.formats.simulation_evaluation_format import (
    manifest_format,
    normalize_input_record,
    parse_simulation,
)


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
            simulation_id = record.get("simulation_id")
            if simulation_id:
                completed.add(str(simulation_id))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/test_inputs/board_simulation.caption.jsonl"))
    parser.add_argument("--prompt", type=Path,
                        default=Path("evaluation/prompts/board_simulation_caption.txt"))
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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
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
    manifest = [normalize_input_record(record, system_prompt) for record in raw_manifest]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    existing = completed_ids(responses_path)
    pending = [record for record in manifest if record["simulation_id"] not in existing]
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
        "task_name": "tumor_board_simulation",
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
        "generation": generation,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "manifest_sha256",
            "manifest_format",
            "model",
            "model_revision",
            "prompt_sha256",
            "parameter_profile",
            "response_mode",
            "generation",
            "provider",
            "base_url",
            "prompt_adapter",
        ):
            previous_value = previous.get(key)
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
                return {"kind": "stopped", "simulation_id": record["simulation_id"]}
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
                        parse_warnings = [final_response_extraction]
                    else:
                        parsed, parse_status, parse_warnings = parse_simulation(
                            extracted_final_response
                        )
                        if parse_status == "parsed":
                            parse_status = "parsed_after_cleanup"
                else:
                    parsed, parse_status, parse_warnings = parse_simulation(content)
                run_key_payload = "\x1f".join(
                    (
                        str(record["benchmark_version"]),
                        str(record["simulation_id"]),
                        args.resolved_model,
                        args.prompt_version,
                        args.parameter_profile,
                    )
                )
                return {
                    "kind": "response",
                    "schema_version": 1,
                    "task_name": "tumor_board_simulation",
                    "run_key": sha256_bytes(run_key_payload.encode("utf-8")),
                    "benchmark_version": record["benchmark_version"],
                    "simulation_id": record["simulation_id"],
                    "video_uid": record["video_uid"],
                    "case_id": record["case_id"],
                    "input_format": input_format,
                    "provider": args.provider,
                    "prompt_adapter": args.prompt_adapter,
                    "requested_model": args.model,
                    "resolved_model": args.resolved_model,
                    "model_revision": args.model_revision,
                    "prompt_version": args.prompt_version,
                    "parameter_profile": args.parameter_profile,
                    "response_mode": args.response_mode,
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
                    "parse_warnings": parse_warnings,
                    "response_id": response_payload.get("id"),
                    "response_model": response_payload.get("model"),
                    "system_fingerprint": response_payload.get("system_fingerprint"),
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:4000]
                last_error = f"HTTPError {exc.code}: {body}"
                if not (exc.code == 429 or 500 <= exc.code <= 599):
                    break
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            if attempt < args.max_retries:
                time.sleep(min(30.0, 2 ** (attempt - 1) + random.random()))
        return {
            "kind": "error",
            "schema_version": 1,
            "simulation_id": record["simulation_id"],
            "failed_utc": utc_now(),
            "attempts": args.max_retries,
            "error": last_error,
        }

    counters = {
        "selected": len(manifest),
        "resumed": len(existing),
        "completed_this_process": 0,
        "parsed": 0,
        "parsed_after_cleanup": 0,
        "parsed_with_warnings": 0,
        "raw_fallback": 0,
        "errors": 0,
    }
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    process_started = time.monotonic()

    def write_summary(final: bool = False) -> None:
        elapsed = time.monotonic() - process_started
        done_total = counters["resumed"] + counters["completed_this_process"]
        rate = counters["completed_this_process"] / elapsed if elapsed else 0.0
        remaining = max(0, counters["selected"] - done_total)
        atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "task_name": "tumor_board_simulation",
                "run_id": output_dir.name,
                "updated_utc": utc_now(),
                "final": final,
                **counters,
                "completed_total": done_total,
                "remaining": remaining,
                "elapsed_seconds_this_process": elapsed,
                "responses_per_second_this_process": rate,
                "usage_this_process": usage,
                "stop_requested": stop.is_set(),
            },
        )

    if not pending:
        write_summary(final=True)
        print("No pending records; run already complete.", flush=True)
        return

    response_handle = responses_path.open("a", encoding="utf-8", buffering=1)
    error_handle = errors_path.open("a", encoding="utf-8", buffering=1)
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency, thread_name_prefix="simulation"
        ) as executor:
            futures = {executor.submit(evaluate, record) for record in pending}
            for future in concurrent.futures.as_completed(futures):
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
                if stop.is_set():
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                write_summary()
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
