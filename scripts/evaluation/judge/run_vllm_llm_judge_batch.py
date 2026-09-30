#!/usr/bin/env python3
"""Run a frozen LLM-judge batch against a local vLLM chat API.

The builder emits OpenAI Responses-style Batch JSONL so that the blind inputs,
schemas, and request IDs can be frozen independently of the execution backend.
This runner translates only the wire format to chat completions.  It preserves
the input messages and JSON Schema verbatim, fixes the local greedy parameters,
disables thinking through the chat template, and writes a Batch-compatible
response envelope after every completed request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = int((len(ordered) - 1) * fraction)
    return ordered[rank]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="default: the batch path with .batch replaced by .output")
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible endpoint serving the judge")
    parser.add_argument("--model", default="judge")
    parser.add_argument("--resolved-model", help="default: --model")
    parser.add_argument("--model-revision", default="unspecified")
    parser.add_argument("--scope", default="test")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--structured-output-whitespace",
        choices=("allow", "disabled"),
        default="disabled",
        help=(
            "Record compact structured output. The vLLM server must use xgrammar "
            "or guidance with disable_any_whitespace=true."
        ),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def response_format(body: dict[str, Any]) -> dict[str, Any]:
    source = body["text"]["format"]
    if source.get("type") != "json_schema":
        raise ValueError("batch request does not use a JSON Schema response format")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": source["name"],
            "strict": bool(source.get("strict")),
            "schema": source["schema"],
        },
    }


def chat_payload(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if row.get("method") != "POST" or row.get("url") != "/v1/responses":
        raise ValueError("unexpected frozen Batch request method or URL")
    body = row["body"]
    messages = body["input"]
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
    ]:
        raise ValueError("judge request must contain exactly system then user")
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": int(body["max_output_tokens"]),
        "response_format": response_format(body),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return payload


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for row in load_jsonl(path):
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in result:
            raise ValueError(f"missing or duplicate custom_id in {path}")
        result.add(custom_id)
    return result


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.batch.with_name(args.batch.name.replace(".batch", ".output"))
    if args.resolved_model is None:
        args.resolved_model = args.model
    if args.concurrency < 1 or args.max_retries < 1:
        raise SystemExit("concurrency and max-retries must be positive")
    if (args.temperature, args.top_p, args.seed) != (0.0, 1.0, 0):
        raise SystemExit("the frozen judge protocol requires temperature=0, top_p=1, seed=0")

    rows = load_jsonl(args.batch)
    all_ids = [str(row.get("custom_id") or "") for row in rows]
    if any(not value for value in all_ids) or len(set(all_ids)) != len(all_ids):
        raise SystemExit("batch has missing or duplicate custom_id values")
    # Validate every frozen request before any inference is sent.
    payloads = {row["custom_id"]: chat_payload(row, args) for row in rows}
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be nonnegative")
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = completed_ids(args.output)
    selected_ids = {str(row["custom_id"]) for row in rows}
    unexpected = existing - selected_ids
    if unexpected:
        raise SystemExit(f"output contains IDs outside selected batch: {sorted(unexpected)[:3]}")
    pending = [row for row in rows if str(row["custom_id"]) not in existing]

    stem = args.output.name[:-6] if args.output.name.endswith(".jsonl") else args.output.name
    config_path = args.output.parent / f"{stem}.local_run_config.json"
    summary_path = args.output.parent / f"{stem}.local_summary.json"
    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "thinking": "disabled",
        "chat_template_kwargs": {"enable_thinking": False},
        "max_output_tokens": sorted(
            {int(row["body"]["max_output_tokens"]) for row in rows}
        ),
        "structured_output": "strict_json_schema",
        "structured_output_whitespace": args.structured_output_whitespace,
    }
    run_config = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "scope": args.scope,
        "batch": str(args.batch.resolve()),
        "batch_sha256": sha256_file(args.batch),
        "selected_requests": len(rows),
        "resumed_requests": len(existing),
        "base_url": args.base_url,
        "endpoint": "/chat/completions",
        "requested_model": args.model,
        "resolved_model": args.resolved_model,
        "model_revision": args.model_revision,
        "generation": generation,
        "wire_translation": {
            "source": "frozen Responses Batch request",
            "target": "local vLLM chat completions",
            "messages_changed": False,
            "json_schema_changed": False,
            "source_reasoning_field_ignored": True,
        },
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "batch_sha256",
            "base_url",
            "requested_model",
            "resolved_model",
            "model_revision",
            "scope",
            "generation",
        ):
            if previous.get(key) != run_config.get(key):
                raise SystemExit(f"refusing incompatible resume: {key} changed")
        run_config["initial_started_utc"] = previous.get(
            "initial_started_utc", previous.get("started_utc")
        )
    atomic_json(config_path, run_config)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        custom_id = str(row["custom_id"])
        request_data = json.dumps(
            payloads[custom_id], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        last_error: dict[str, Any] | None = None
        for attempt in range(1, args.max_retries + 1):
            if stop.is_set():
                return {"kind": "stopped", "custom_id": custom_id}
            started_perf = time.perf_counter()
            started_utc = utc_now()
            request = urllib.request.Request(
                endpoint,
                data=request_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    raw = json.load(response)
                    request_id = response.headers.get("x-request-id") or raw.get("id")
                    status_code = response.status
                elapsed = time.perf_counter() - started_perf
                choice = raw["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("chat completion content is not a string")
                body = {
                    "id": raw.get("id"),
                    "object": "response",
                    "created_at": raw.get("created"),
                    "status": "completed",
                    "model": raw.get("model"),
                    "output": [
                        {
                            "id": f"msg_{raw.get('id')}",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content,
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": raw.get("usage"),
                    "local_chat_completion": raw,
                }
                return {
                    "kind": "result",
                    "row": {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": status_code,
                            "request_id": request_id,
                            "body": body,
                        },
                        "error": None,
                        "local_execution": {
                            "attempt": attempt,
                            "started_utc": started_utc,
                            "finished_utc": utc_now(),
                            "latency_seconds": elapsed,
                            "finish_reason": choice.get("finish_reason"),
                            "thinking_disabled": True,
                        },
                    },
                }
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")[:12000]
                try:
                    body_value: Any = json.loads(body_text)
                except json.JSONDecodeError:
                    body_value = body_text
                last_error = {
                    "type": "http_error",
                    "status_code": exc.code,
                    "message": body_value,
                }
                retryable = exc.code == 429 or 500 <= exc.code <= 599
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = {"type": type(exc).__name__, "message": str(exc)}
                retryable = True
            except Exception as exc:
                last_error = {"type": type(exc).__name__, "message": str(exc)}
                retryable = False
            if not retryable or attempt == args.max_retries:
                break
            time.sleep(min(30.0, float(2 ** (attempt - 1))))
        return {
            "kind": "result",
            "row": {
                "custom_id": custom_id,
                "response": {
                    "status_code": (last_error or {}).get("status_code", 0),
                    "request_id": None,
                    "body": None,
                },
                "error": last_error,
                "local_execution": {
                    "attempt": attempt,
                    "started_utc": started_utc,
                    "finished_utc": utc_now(),
                    "latency_seconds": time.perf_counter() - started_perf,
                    "finish_reason": None,
                    "thinking_disabled": True,
                },
            },
        }

    process_started = time.monotonic()
    completed_this_process = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [executor.submit(evaluate, row) for row in pending]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["kind"] == "stopped":
                    continue
                output_handle.write(
                    json.dumps(result["row"], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
                completed_this_process += 1
                if completed_this_process % 10 == 0 or completed_this_process == len(pending):
                    elapsed = time.monotonic() - process_started
                    print(
                        json.dumps(
                            {
                                "completed_this_process": completed_this_process,
                                "completed_total": len(existing) + completed_this_process,
                                "selected": len(rows),
                                "remaining": len(rows)
                                - len(existing)
                                - completed_this_process,
                                "requests_per_second": completed_this_process / elapsed
                                if elapsed
                                else 0.0,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

    output_rows = load_jsonl(args.output)
    output_by_id = {str(row["custom_id"]): row for row in output_rows}
    selected_output = [output_by_id[value] for value in all_ids[: len(rows)]]
    latencies = [
        float(row["local_execution"]["latency_seconds"])
        for row in selected_output
        if row.get("local_execution", {}).get("latency_seconds") is not None
    ]
    successful = [
        row
        for row in selected_output
        if row.get("error") is None and row.get("response", {}).get("status_code") == 200
    ]
    finish_reasons = Counter(
        str(row.get("local_execution", {}).get("finish_reason")) for row in successful
    )
    usage = Counter()
    for row in successful:
        raw_usage = row["response"]["body"].get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(raw_usage.get(key) or 0)
    elapsed = time.monotonic() - process_started
    summary = {
        "schema_version": 1,
        "completed_utc": utc_now(),
        "selected": len(rows),
        "completed_total": len(selected_output),
        "successful": len(successful),
        "errors": len(selected_output) - len(successful),
        "remaining": len(rows) - len(selected_output),
        "finish_reasons": dict(finish_reasons),
        "usage": dict(usage),
        "latency_seconds": {
            "mean": fmean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "process_elapsed_seconds": elapsed,
        "process_throughput_requests_per_second": completed_this_process / elapsed
        if elapsed
        else 0.0,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "run_config": str(config_path.resolve()),
        "run_config_sha256": sha256_file(config_path),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
