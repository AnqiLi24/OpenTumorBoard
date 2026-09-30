#!/usr/bin/env python3
"""Input adapters, output parser and conclusion extractor for Board Simulation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


USER_TEMPLATE = """CASE SUMMARY:
{case_summary}

SLIDE DESCRIPTIONS:
{slides}"""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def messages_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical_json(messages)).hexdigest()


def format_user_prompt(record: dict[str, Any]) -> str:
    return USER_TEMPLATE.format(
        case_summary=record["case_summary"],
        slides=record["slides"] or "No slide descriptions are available.",
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
    simulation_id = str(metadata.get("simulation_id") or "")
    if not simulation_id or str(record.get("id") or "") != simulation_id:
        raise ValueError("ShareGPT id must match metadata.simulation_id")
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
            "ShareGPT conversations do not match the frozen prompt for "
            f"{metadata['simulation_id']}"
        )
    actual_hash = messages_sha256(expected_messages)
    declared_hash = record.get("messages_sha256")
    if declared_hash is not None and declared_hash != actual_hash:
        raise ValueError(
            f"ShareGPT messages_sha256 mismatch for {metadata['simulation_id']}"
        )
    metadata["_messages"] = expected_messages
    return metadata


def input_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Remove hidden references before producing a model-input artifact."""
    return {
        key: value
        for key, value in record_metadata(record).items()
        if not key.startswith("_") and not key.startswith("reference_")
        and key not in {"input_sha256", "source_file_sha256", "messages_sha256", "prompt_sha256"}
    }


def make_sharegpt_record(
    record: dict[str, Any],
    system_prompt: str,
    prompt_version: str,
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = input_metadata(record)
    messages = build_openai_messages(metadata, system_prompt)
    conversations: list[dict[str, str]] = [
        {"from": "system", "value": messages[0]["content"]},
        {"from": "human", "value": messages[1]["content"]},
    ]
    output: dict[str, Any] = {
        "schema_version": 1,
        "format": "sharegpt",
        "id": metadata["simulation_id"],
        "prompt_version": prompt_version,
        "conversations": conversations,
        "metadata": metadata,
    }
    if response is not None:
        raw_response = response.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError(
                f"Response for {metadata['simulation_id']} has no raw response"
            )
        conversations.append({"from": "gpt", "value": raw_response})
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
                "finish_reason",
                "parse_status",
            )
        }
    return output


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)", text.strip()))


#
_TURN_HEAD = re.compile(r"(?m)^\s*<\s*turn\s+(\d+)\s*>\s*([^:\n]{1,80}?)\s*:\s*(.*)$")


def _looks_like_turn_format(text: str) -> bool:
    return _TURN_HEAD.search(text) is not None


def _parse_turn_transcript(cleaned: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    warnings: list[str] = []
    open_disc = re.search(r"<\s*discussion\s*>", cleaned, re.IGNORECASE)
    close_disc = re.search(r"<\s*/\s*discussion\s*>", cleaned, re.IGNORECASE)
    body_start = open_disc.end() if open_disc else 0
    body_end = close_disc.start() if close_disc else len(cleaned)
    if open_disc is None:
        warnings.append("missing <discussion> tag")
    if close_disc is None:
        warnings.append("missing </discussion> tag")

    conc_open = re.search(r"<\s*conclusion\s*>", cleaned, re.IGNORECASE)
    if conc_open is None:
        return None, "raw_fallback", ["missing <conclusion> section"]
    conc_close = re.search(r"<\s*/\s*conclusion\s*>", cleaned[conc_open.end():], re.IGNORECASE)
    conclusion = (cleaned[conc_open.end():conc_open.end() + conc_close.start()]
                  if conc_close else cleaned[conc_open.end():]).strip()
    if conc_close is None:
        warnings.append("missing </conclusion> tag")
    if close_disc is None:
        body_end = min(body_end, conc_open.start())

    body = cleaned[body_start:body_end]
    heads = list(_TURN_HEAD.finditer(body))
    rows: list[dict[str, str]] = []
    numbers: list[int] = []
    for index, head in enumerate(heads):
        stop = heads[index + 1].start() if index + 1 < len(heads) else len(body)
        text = (head.group(3) + "\n" + body[head.end():stop]).strip()
        if not text:
            continue
        numbers.append(int(head.group(1)))
        rows.append({"role": head.group(2).strip().strip("*"), "response": text})

    if not rows or not conclusion:
        return None, "raw_fallback", ["missing turns or conclusion text"]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))) or numbers[0] != 1:
        warnings.append("turn numbers are not 1..N in order")

    normalized_roles = [row["role"].casefold() for row in rows]
    if any("insufficient_evidence" in role for role in normalized_roles):
        warnings.append("disallowed insufficient_evidence role")
    sentence_count = _sentence_count(conclusion)
    if not 2 <= sentence_count <= 4:
        warnings.append(
            f"expected a 2-4 sentence conclusion, found approximately {sentence_count}"
        )
    parsed = {
        "discussion": rows,
        "conclusion": conclusion,
        "discussion_row_count": len(rows),
        "unique_role_count": len(set(normalized_roles)),
        "conclusion_sentence_count": sentence_count,
    }
    return parsed, ("parsed_with_warnings" if warnings else "parsed"), warnings


def parse_simulation(content: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Parse the required table while always preserving raw model text elsewhere."""
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
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if _looks_like_turn_format(cleaned):
        parsed, status, warnings = _parse_turn_transcript(cleaned)
        if parsed is not None and removed_thinking and status == "parsed":
            status = "parsed_after_cleanup"
        return parsed, status, warnings

    match = re.search(
        r"(?im)^\s*(?:#{1,6}\s*)?"
        r"(?:\*\*conclusion:\*\*|\*\*conclusion\*\*:|conclusion:)\s*(.*)$",
        cleaned,
    )
    if not match:
        return None, "raw_fallback", ["missing Conclusion section"]
    table_text = cleaned[: match.start()].strip()
    conclusion = (match.group(1) + "\n" + cleaned[match.end() :]).strip()
    rows: list[dict[str, str]] = []
    current_role: str | None = None
    current_response: list[str] = []

    def finish_row() -> None:
        nonlocal current_role, current_response
        if current_role is None:
            return
        response = "\n".join(current_response).strip()
        if response.endswith("|"):
            response = response[:-1].rstrip()
        if response:
            rows.append({"role": current_role, "response": response})
        current_role = None
        current_response = []

    for line in table_text.splitlines():
        row_match = re.match(r"^\s*\|\s*([^|]+?)\s*\|\s*(.*)$", line)
        if row_match:
            role = row_match.group(1).strip()
            response = row_match.group(2).strip()
            if role.casefold() == "role" or re.fullmatch(r":?-+:?", role):
                continue
            finish_row()
            current_role = role.strip("*").strip()
            current_response = [response]
        elif current_role is not None:
            current_response.append(line.rstrip())
    finish_row()
    if not rows or not conclusion:
        return None, "raw_fallback", ["missing table rows or conclusion text"]

    warnings: list[str] = []
    normalized_roles = [row["role"].casefold() for row in rows]
    if len(set(normalized_roles)) != len(normalized_roles):
        warnings.append("duplicate specialist roles")
    if any("insufficient_evidence" in role for role in normalized_roles):
        warnings.append("disallowed insufficient_evidence role")
    sentence_count = _sentence_count(conclusion)
    if not 2 <= sentence_count <= 4:
        warnings.append(
            f"expected a 2-4 sentence conclusion, found approximately {sentence_count}"
        )
    parsed = {
        "discussion": rows,
        "conclusion": conclusion,
        "discussion_row_count": len(rows),
        "unique_role_count": len(set(normalized_roles)),
        "conclusion_sentence_count": sentence_count,
    }
    if warnings:
        status = "parsed_with_warnings"
    elif removed_thinking:
        status = "parsed_after_cleanup"
    else:
        status = "parsed"
    return parsed, status, warnings


# A response that is nothing but a labelled conclusion, as a model finetuned on
# conclusions alone produces. parse_simulation needs discussion rows as well.
CONCLUSION_ONLY_RE = re.compile(
    r"\A\s*(?:#{1,6}\s*)?"
    r"(?:\*\*conclusion:\*\*|\*\*conclusion\*\*:|conclusion:)\s*(.+)\Z",
    re.IGNORECASE | re.DOTALL,
)
XML_CONCLUSION_CLOSED = re.compile(
    r"<\s*conclusion\s*>(.*?)<\s*/\s*conclusion\s*>", re.IGNORECASE | re.DOTALL
)
# A response cut at max_tokens keeps the opening tag and loses the closing one.
XML_CONCLUSION_OPEN = re.compile(r"<\s*conclusion\s*>(.+)\Z", re.IGNORECASE | re.DOTALL)


def extract_conclusion(response: dict[str, Any]) -> tuple[str, str]:
    """Return the conclusion of a Board Simulation response and how it was found.

    The judge, ROUGE-L and BERTScore all score this text. An empty string means
    the response has no conclusion; it is a format failure and is scored at the
    floor of the scale.
    """
    parsed = response.get("parsed_output")
    if isinstance(parsed, dict) and isinstance(parsed.get("conclusion"), str):
        conclusion = parsed["conclusion"].strip()
        if conclusion:
            return conclusion, "parsed"
    method = "unavailable"
    direct_final_without_trace = False
    if response.get("response_mode") == "reasoning_final":
        text = response.get("extracted_final_response")
        if not isinstance(text, str) or not text.strip():
            candidate = response.get("raw_response")
            # A reasoning run whose answer has no explicit final boundary is used
            # only when it carries no reasoning trace and opens with the table.
            if (
                response.get("raw_reasoning_content") is None
                and response.get("final_response_extraction")
                == "missing_explicit_final_boundary"
                and isinstance(candidate, str)
                and candidate.lstrip().startswith("|")
            ):
                text = candidate
                direct_final_without_trace = True
    else:
        text = response.get("raw_response")
    if isinstance(text, str) and text.strip():
        reparsed, _, _ = parse_simulation(text)
        if isinstance(reparsed, dict):
            conclusion = str(reparsed.get("conclusion") or "").strip()
            if conclusion:
                return conclusion, (
                    "direct_final_markdown_no_reasoning_trace"
                    if direct_final_without_trace
                    else "raw_markdown_conclusion"
                )
    text = response.get("extracted_final_response") or response.get("raw_response") or ""
    if not isinstance(text, str):
        return "", method
    if "|" not in text:
        match = CONCLUSION_ONLY_RE.match(text.strip())
        if match and match.group(1).strip():
            return match.group(1).strip(), "conclusion_only_response"
    for pattern in (XML_CONCLUSION_CLOSED, XML_CONCLUSION_OPEN):
        match = pattern.search(text)
        if match and match.group(1).strip():
            return match.group(1).strip(), "conclusion_xml"
    return "", method
