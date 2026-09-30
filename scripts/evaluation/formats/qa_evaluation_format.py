#!/usr/bin/env python3
"""Lossless custom-manifest and ShareGPT adapters for QA evaluation."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


USER_TEMPLATE = """TARGET SPECIALIST ROLE:
{target_specialist_role}

CASE SUMMARY:
{case_summary}

SLIDE DESCRIPTIONS:
{slides}

QUESTION:
{question}"""

ANSWER_WRAPPER_RE = re.compile(
    r'^\s*\{\s*"answer"\s*:\s*"', re.IGNORECASE | re.DOTALL
)
JSON_ESCAPE_RE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _decode_json_string_fragment(value: str) -> str:
    """Decode only JSON string escapes, without rewriting answer wording."""
    simple = {
        r'\"': '"',
        r'\\': '\\',
        r'\/': '/',
        r'\b': '\b',
        r'\f': '\f',
        r'\n': '\n',
        r'\r': '\r',
        r'\t': '\t',
    }

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(r'\u'):
            return chr(int(token[2:], 16))
        return simple[token]

    return JSON_ESCAPE_RE.sub(replace, value)


def extract_answer_text(response: dict[str, Any]) -> tuple[str | None, str]:
    """Extract answer content without silently repairing clinical language.

    The second return value records the exact deterministic path used. A raw
    JSON-like response may be missing only its closing brace/quote; removing
    that transport wrapper is allowed, while generating or paraphrasing text is
    not. Plain-text fallbacks are already the model's answer content.
    """
    parsed = response.get("parsed_output")
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        answer = parsed["answer"].strip()
        return (answer, "parsed") if answer else (None, "unrecoverable")

    direct_final_without_trace = False
    if response.get("response_mode") == "reasoning_final":
        raw = response.get("extracted_final_response")
        if not isinstance(raw, str) or not raw.strip():
            if (
                response.get("raw_reasoning_content") is None
                and response.get("final_response_extraction")
                == "missing_explicit_final_boundary"
            ):
                raw = response.get("raw_response")
                direct_final_without_trace = True
            else:
                return None, "unrecoverable_reasoning_final"
    else:
        raw = response.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return None, "unrecoverable"
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        decoded = json.loads(cleaned)
    except json.JSONDecodeError:
        decoded = None
    if (
        isinstance(decoded, dict)
        and isinstance(decoded.get("answer"), str)
        and (not direct_final_without_trace or set(decoded) == {"answer"})
    ):
        answer = decoded["answer"].strip()
        if not answer:
            return None, "unrecoverable"
        method = (
            "direct_final_json_no_reasoning_trace"
            if direct_final_without_trace
            else "raw_json"
        )
        return answer, method

    # A reasoning-mode response with no visible trace is scoreable only when
    # its entire raw content is the strict requested JSON object. This recovers
    # provider/checkpoint combinations that emit the final directly without
    # ever treating unfinished prose reasoning as an answer.
    if direct_final_without_trace:
        return None, "unrecoverable_reasoning_final"

    wrapper = ANSWER_WRAPPER_RE.match(cleaned)
    if wrapper:
        answer = cleaned[wrapper.end():].strip()
        if answer.endswith("}"):
            answer = answer[:-1].rstrip()
        if answer.endswith('"'):
            answer = answer[:-1]
        answer = _decode_json_string_fragment(answer).strip()
        return (
            (answer, "raw_json_wrapper_recovered")
            if answer
            else (None, "unrecoverable")
        )

    # A JSON-shaped payload with no extractable `answer` is a format failure,
    # not plain clinical prose. Never send its syntax into a semantic metric.
    if cleaned.startswith(("{", "[")):
        return None, "unrecoverable"

    # The model omitted the requested JSON envelope, but the raw response is
    # otherwise exactly the answer. Preserve it verbatim for semantic scoring.
    if cleaned not in {"{}", "[]", '""'}:
        return cleaned, "raw_plain_text"
    return None, "unrecoverable"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def messages_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical_json(messages)).hexdigest()


def format_user_prompt(record: dict[str, Any]) -> str:
    return USER_TEMPLATE.format(
        target_specialist_role=record["target_specialist_role"],
        case_summary=record["case_summary"],
        slides=record["slides"],
        question=record["question"],
    )


def build_openai_messages(
    record: dict[str, Any], system_prompt: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_user_prompt(record)},
    ]


def manifest_format(records: list[dict[str, Any]]) -> str:
    formats = {
        "sharegpt" if isinstance(record.get("conversations"), list) else "custom"
        for record in records
    }
    if not formats:
        raise ValueError("manifest is empty")
    if len(formats) != 1:
        raise ValueError("manifest mixes custom and ShareGPT records")
    return formats.pop()


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record.get("conversations"), list):
        return record
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ShareGPT record metadata must be an object")
    qa_id = str(metadata.get("qa_id") or "")
    if not qa_id or str(record.get("id") or "") != qa_id:
        raise ValueError("ShareGPT id must match metadata.qa_id")
    return metadata


def normalize_input_record(
    record: dict[str, Any], system_prompt: str
) -> dict[str, Any]:
    metadata = dict(record_metadata(record))
    expected_messages = build_openai_messages(metadata, system_prompt)
    if not isinstance(record.get("conversations"), list):
        metadata["_messages"] = expected_messages
        return metadata

    conversations = record["conversations"]
    if len(conversations) != 2:
        raise ValueError(
            "ShareGPT inference records must contain exactly system and human turns"
        )
    expected_turns = [
        {"from": "system", "value": expected_messages[0]["content"]},
        {"from": "human", "value": expected_messages[1]["content"]},
    ]
    if conversations != expected_turns:
        raise ValueError(
            f"ShareGPT conversations do not match the frozen prompt for {metadata['qa_id']}"
        )
    declared_hash = record.get("messages_sha256")
    actual_hash = messages_sha256(expected_messages)
    if declared_hash is not None and declared_hash != actual_hash:
        raise ValueError(
            f"ShareGPT messages_sha256 mismatch for {metadata['qa_id']}"
        )
    metadata["_messages"] = expected_messages
    return metadata


MULTIMODAL_MARKER = "<image>"


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# `<image>` only marks a slide inside the slide block; elsewhere, for example in a
# candidate that echoes the prompt's example, it is ordinary text.
SLIDE_BLOCK_RE = re.compile(
    r"(<SLIDE_DESCRIPTIONS>\n)(.*?)(\n</SLIDE_DESCRIPTIONS>)", re.DOTALL
)


def build_multimodal_content(human: str, images: list[Path]) -> list[dict[str, Any]]:
    """Interleave text and images so each slide sits beside its own caption.

    Markers are counted inside <SLIDE_DESCRIPTIONS> when that block is present,
    and over the whole string otherwise - the Task 2 user turn has no such block
    and its caption list is the whole text.
    """
    block = SLIDE_BLOCK_RE.search(human)
    if block is not None:
        head = human[: block.start(2)]
        tail = human[block.end(2) :]
        inner = block.group(2)
        parts = inner.split(MULTIMODAL_MARKER)
        if len(parts) - 1 != len(images):
            raise ValueError(
                f"{len(parts) - 1} markers in the slide block but {len(images)} images"
            )
        if MULTIMODAL_MARKER.join(parts) != inner:
            raise ValueError("splitting on the marker did not round-trip")
        parts = [head + parts[0]] + parts[1:-1] + [parts[-1] + tail] if len(parts) > 1 \
            else [head + parts[0] + tail]
        content: list[dict[str, Any]] = []
        for index, chunk in enumerate(parts):
            if chunk:
                content.append({"type": "text", "text": chunk})
            if index < len(images):
                content.append(
                    {"type": "image_url",
                     "image_url": {"url": _data_uri(images[index])}}
                )
        return content
    parts = human.split(MULTIMODAL_MARKER)
    if len(parts) - 1 != len(images):
        raise ValueError(f"{len(parts) - 1} markers but {len(images)} images")
    if MULTIMODAL_MARKER.join(parts) != human:
        raise ValueError("splitting on the marker did not round-trip")
    content: list[dict[str, Any]] = []
    for index, chunk in enumerate(parts):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if index < len(images):
            content.append(
                {"type": "image_url", "image_url": {"url": _data_uri(images[index])}}
            )
    return content


def normalize_multimodal_record(
    record: dict[str, Any], system_prompt: str
) -> dict[str, Any]:
    """The multimodal sibling of normalize_input_record.

    The multimodal turn names its slide block `SLIDES:` rather than
    `SLIDE DESCRIPTIONS:`, so the human turn is checked instead of rebuilt: the
    system turn is the prompt file, the turns are exactly system+human, and every
    marker has an existing image file.
    """
    metadata = dict(record_metadata(record))
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{metadata.get('qa_id')}: expected system and human turns")
    if [turn["from"] for turn in conversations] != ["system", "human"]:
        raise ValueError(f"{metadata.get('qa_id')}: turns are not system+human")
    if conversations[0]["value"].strip() != system_prompt.strip():
        raise ValueError(f"{metadata.get('qa_id')}: system turn is not the prompt file")

    human = conversations[1]["value"]
    images = [Path(p) for p in (record.get("images") or [])]
    missing = [str(p) for p in images if not p.is_file()]
    if missing:
        raise ValueError(f"{metadata.get('qa_id')}: missing images {missing[:2]}")

    text_messages = [
        {"role": "system", "content": conversations[0]["value"]},
        {"role": "user", "content": human},
    ]
    declared = record.get("messages_sha256")
    actual = messages_sha256(text_messages)
    if declared is not None and declared != actual:
        raise ValueError(f"{metadata.get('qa_id')}: messages_sha256 mismatch")

    metadata["_images"] = [str(p) for p in images]
    metadata["_messages"] = [
        {"role": "system", "content": conversations[0]["value"]},
        {"role": "user", "content": build_multimodal_content(human, images)},
    ]
    return metadata


def make_sharegpt_record(
    record: dict[str, Any],
    system_prompt: str,
    prompt_version: str,
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in record_metadata(record).items()
        if not key.startswith("_")
        and key not in {"input_sha256", "source_file_sha256", "messages_sha256", "prompt_sha256"}
    }
    messages = build_openai_messages(metadata, system_prompt)
    conversations: list[dict[str, str]] = [
        {"from": "system", "value": messages[0]["content"]},
        {"from": "human", "value": messages[1]["content"]},
    ]
    output: dict[str, Any] = {
        "schema_version": 1,
        "format": "sharegpt",
        "id": metadata["qa_id"],
        "prompt_version": prompt_version,
        "conversations": conversations,
        "metadata": metadata,
    }
    if response is not None:
        parsed = response.get("parsed_output")
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            gpt_value = parsed["answer"]
        elif (
            response.get("parse_status") == "raw_fallback"
            and isinstance(response.get("raw_response"), str)
        ):
            # Preserve schema-invalid model output losslessly instead of dropping
            # the record or selectively regenerating it.
            gpt_value = response["raw_response"]
        else:
            raise ValueError(
                f"Response for {metadata['qa_id']} has neither a parsed answer "
                "nor a raw fallback"
            )
        conversations.append({"from": "gpt", "value": gpt_value})
        output["generation"] = {
            key: response.get(key)
            for key in (
                "run_key",
                "provider",
                "requested_model",
                "resolved_model",
                "model_revision",
                "prompt_version",
                "parameter_profile",
                "output_schema",
                "finish_reason",
                "parse_status",
            )
        }
    return output
