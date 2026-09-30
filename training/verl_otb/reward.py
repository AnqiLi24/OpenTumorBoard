"""Dr.GRPO reward for Board Simulation.

A response is one ``<discussion>`` block of numbered turns followed by a
one-line ``<conclusion>``. The reward combines three judged terms with a
deterministic format term:

    support = decisions * consistency
    match   = 0.40 * alignment + 0.35 * support + 0.25 * decisions
    reward  = match + 0.10 * format_score

``alignment`` is the conclusion-alignment rubric on the candidate conclusion,
``decisions`` the weighted mean of the four-decision rubric over the decisions
the reference conclusion actually takes, and ``consistency`` asks a third judge
call whether the candidate's own discussion developed the conclusion it
reached. The decisions in scope are frozen before training by
``precompute_reference_scope.py``, so no candidate wording can add or remove a
decision from the denominator.

A malformed, truncated or severely repetitive response skips every judge call
and takes ``match = -0.20``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ALIGNMENT_PROMPT_PATH = HERE / "source_data" / "conclusion_alignment.txt"
DECISIONS_PROMPT_PATH = HERE / "source_data" / "four_decisions.txt"
CONSISTENCY_PROMPT_PATH = HERE / "source_data" / "consistency.txt"

DIMENSIONS = ("therapy", "surgery", "next_action", "clinical_trial")
DIMENSION_WEIGHTS = {"therapy": 1.00, "surgery": 1.25, "next_action": 1.15, "clinical_trial": 2.00}

ALIGNMENT_WEIGHT = 0.40
SUPPORT_WEIGHT = 0.35
DECISIONS_WEIGHT = 0.25
FORMAT_WEIGHT = 0.10
INVALID_REWARD = -0.20

# Weights inside the two judged aggregates, kept as the training run used them.
ANCHOR_ALIGNMENT_WEIGHT = 0.45
ANCHOR_DECISIONS_WEIGHT = 0.55


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

DISCUSSION_RE = re.compile(r"<discussion>\s*(?P<text>.*?)\s*</discussion>", re.DOTALL)
CONCLUSION_RE = re.compile(r"<conclusion>\s*(?P<text>.*?)\s*</conclusion>", re.DOTALL)
# The reference is written by the data builder and always closes its conclusion
# on one line, so it is matched more strictly than a candidate's.
REFERENCE_DISCUSSION_RE = re.compile(r"<discussion>\s*(?P<text>.*?)\s*</discussion>", re.DOTALL)
REFERENCE_CONCLUSION_RE = re.compile(r"<conclusion>\s*(?P<text>[^<>]+?)\s*</conclusion>", re.DOTALL)
ORDERED_BLOCKS_RE = re.compile(
    r"\A<discussion>\n(?P<discussion>.*?)\n</discussion>\n"
    r"<conclusion>(?P<conclusion>[^\n]+)</conclusion>\Z",
    re.DOTALL,
)
TURN_RE = re.compile(r"^<turn (?P<number>[1-9]\d*)> (?P<role>[^:\n<>]+): (?P<text>.+)$")
SENTENCE_END_RE = re.compile(
    r"(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\b[A-Z])[.!?](?:[\"'”]*)?(?=\s|$)"
)


def _extract_unique(pattern: re.Pattern[str], response: str) -> str | None:
    matches = [match.group("text").strip() for match in pattern.finditer(response or "")]
    return matches[0] if len(matches) == 1 and matches[0] else None


def extract_discussion(response: str) -> str | None:
    return _extract_unique(DISCUSSION_RE, response)


def extract_conclusion(response: str) -> str | None:
    return _extract_unique(CONCLUSION_RE, response)


def _extract_reference_block(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text or "")
    if len(matches) != 1:
        return None
    value = matches[0][0] if isinstance(matches[0], tuple) else matches[0]
    return value.strip()


def extract_reference_parts(ground_truth: str) -> tuple[str, str]:
    discussion = _extract_reference_block(REFERENCE_DISCUSSION_RE, ground_truth)
    conclusion = _extract_reference_block(REFERENCE_CONCLUSION_RE, ground_truth)
    if discussion is None or conclusion is None:
        raise ValueError(
            "ground_truth must contain exactly one complete <discussion> block and one "
            "<conclusion> block; rebuild the parquet files with prepare_data.py"
        )
    return discussion, conclusion


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# Frozen decision scope
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _scope_records() -> dict[tuple[str, str], dict]:
    path_value = os.environ.get("OTB_REFERENCE_SCOPE_PATH")
    if not path_value:
        raise RuntimeError("OTB_REFERENCE_SCOPE_PATH is required")
    records = {}
    for line in Path(path_value).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[(str(record["source_id"]), str(record["reference_sha256"]))] = record
    if not records:
        raise RuntimeError(f"empty reference scope cache: {path_value}")
    return records


def _get_scope(reference: str, extra_info: dict | None) -> dict[str, str]:
    identity = extra_info or {}
    source_id = str(identity.get("source_id") or identity.get("input_sha256"))
    key = (source_id, _sha(reference))
    try:
        scope = _scope_records()[key]["scope"]
    except KeyError as error:
        raise RuntimeError(
            f"reference scope missing for source_id={source_id!r}, sha256={key[1]}"
        ) from error
    if any(scope.get(name) not in {"positive", "negative", "absent"} for name in DIMENSIONS):
        raise ValueError(f"invalid cached scope for {source_id}")
    return scope


# --------------------------------------------------------------------------
# Judge transport and audit log
# --------------------------------------------------------------------------


def _judge_endpoint() -> str:
    return os.environ.get("OTB_JUDGE_BASE_URL", "http://127.0.0.1:8094/v1").rstrip("/")


def _post_json(payload: dict) -> tuple[dict, float]:
    endpoint = _judge_endpoint()
    retries = int(os.environ.get("OTB_JUDGE_RETRIES", "3"))
    timeout = float(os.environ.get("OTB_JUDGE_TIMEOUT", "300"))
    data = json.dumps(payload).encode()
    error: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                endpoint + "/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
            result = json.loads(raw["choices"][0]["message"]["content"])
            return result, time.perf_counter() - started
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"Conclusion judge failed after {retries} attempts: {error}") from error


def _append_audit(record: dict[str, Any]) -> None:
    path_value = os.environ.get("OTB_JUDGE_AUDIT_LOG")
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def _payload(prompt: str, user: str, schema: dict[str, Any], name: str, max_tokens: int) -> dict:
    return {
        "model": os.environ.get("OTB_JUDGE_MODEL", "judge"),
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    }


# --------------------------------------------------------------------------
# Conclusion judges: alignment and the four decisions
# --------------------------------------------------------------------------

ALIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "conclusion_alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 1000},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        }
    },
    "required": ["conclusion_alignment"],
    "additionalProperties": False,
}


def _decisions_schema(scope: dict[str, str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        name: (
            {"type": "integer", "enum": [-1]}
            if scope[name] == "absent"
            else {"type": "integer", "minimum": 0, "maximum": 5}
        )
        for name in DIMENSIONS
    }
    properties["rationale"] = {"type": "string", "minLength": 1, "maxLength": 1200}
    return {
        "type": "object",
        "properties": properties,
        "required": [*DIMENSIONS, "rationale"],
        "additionalProperties": False,
    }


def _evidence_parts(evidence: str) -> tuple[str, str]:
    text = (evidence or "").strip()
    prefix = "CASE SUMMARY:\n"
    separator = "\n\nSLIDES:\n"
    if text.startswith(prefix) and separator in text:
        case_summary, slides = text[len(prefix) :].split(separator, 1)
        return case_summary.strip(), slides.strip() or "No slide descriptions are available."
    # Keep the tagged structure identical to evaluation even when the source
    # carries no split, and let the audit log show the missing separator.
    return text, "No slide descriptions are available."


def _conclusion_user_text(evidence: str, reference: str, candidate: str) -> str:
    case_summary, slides = _evidence_parts(evidence)
    return "\n\n".join(
        (
            f"<CASE_SUMMARY>\n{case_summary}\n</CASE_SUMMARY>",
            f"<SLIDE_DESCRIPTIONS>\n{slides}\n</SLIDE_DESCRIPTIONS>",
            f"<REFERENCE_CONCLUSION>\n{reference}\n</REFERENCE_CONCLUSION>",
            f"<CANDIDATE_RESPONSE>\n{candidate}\n</CANDIDATE_RESPONSE>",
        )
    )


def _validate_alignment(result: dict) -> tuple[int, str]:
    alignment = result["conclusion_alignment"]
    score = alignment["score"]
    rationale = alignment["rationale"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"invalid conclusion_alignment: {score!r}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("empty alignment rationale")
    return score, rationale


def _validate_decisions(result: dict, scope: dict[str, str]) -> tuple[dict[str, int], str]:
    scores = {}
    for name in DIMENSIONS:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value <= 5:
            raise ValueError(f"invalid {name}: {value!r}")
        expected_absent = scope[name] == "absent"
        if expected_absent and value != -1:
            raise ValueError(f"{name} must be -1 under the frozen absent scope")
        if not expected_absent and value < 0:
            raise ValueError(f"{name} must be scored under the frozen present scope")
        scores[name] = value
    rationale = result["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("empty decisions rationale")
    return scores, rationale


def _conclusion_details(
    alignment: dict, decisions: dict, scope: dict[str, str]
) -> tuple[dict[str, float], str, str]:
    alignment_score, alignment_rationale = _validate_alignment(alignment)
    dimension_scores, decisions_rationale = _validate_decisions(decisions, scope)
    alignment_normalized = (alignment_score - 1) / 4.0
    numerator = denominator = 0.0
    for name in DIMENSIONS:
        if scope[name] != "absent":
            weight = DIMENSION_WEIGHTS[name]
            numerator += weight * dimension_scores[name] / 5.0
            denominator += weight
    decisions_normalized = numerator / denominator if denominator else 0.0
    match = (
        ANCHOR_ALIGNMENT_WEIGHT * alignment_normalized
        + ANCHOR_DECISIONS_WEIGHT * decisions_normalized
        if denominator
        else alignment_normalized
    )
    details = {
        "alignment_score": float(alignment_score),
        "alignment_normalized": alignment_normalized,
        **{f"decision_{name}_score": float(dimension_scores[name]) for name in DIMENSIONS},
        "applicable_decision_count": float(sum(scope[name] != "absent" for name in DIMENSIONS)),
        "decisions_normalized": decisions_normalized,
        "match_score": match,
        "judge_called": 1.0,
    }
    return details, alignment_rationale, decisions_rationale


def judge_conclusion(candidate: str, reference: str, extra_info: dict | None = None) -> dict[str, float]:
    """Score the candidate conclusion on both conclusion rubrics."""

    identity = extra_info or {}
    scope = _get_scope(reference, identity)
    user = _conclusion_user_text(str(identity.get("case_evidence") or ""), reference, candidate)
    alignment_payload = _payload(
        ALIGNMENT_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        user,
        ALIGNMENT_SCHEMA,
        "otb_conclusion_alignment",
        int(os.environ.get("OTB_ALIGNMENT_MAX_TOKENS", "512")),
    )
    decisions_payload = _payload(
        DECISIONS_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        user,
        _decisions_schema(scope),
        "otb_four_decisions",
        int(os.environ.get("OTB_DECISIONS_MAX_TOKENS", "768")),
    )
    started = time.perf_counter()
    if os.environ.get("OTB_PARALLEL_JUDGES", "1") == "1":
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            alignment_future = pool.submit(_post_json, alignment_payload)
            decisions_future = pool.submit(_post_json, decisions_payload)
            alignment, alignment_latency = alignment_future.result()
            decisions, decisions_latency = decisions_future.result()
    else:
        alignment, alignment_latency = _post_json(alignment_payload)
        decisions, decisions_latency = _post_json(decisions_payload)
    details, alignment_rationale, decisions_rationale = _conclusion_details(
        alignment, decisions, scope
    )
    details.update(
        {
            "judge_latency_seconds": time.perf_counter() - started,
            "alignment_judge_latency_seconds": alignment_latency,
            "decisions_judge_latency_seconds": decisions_latency,
        }
    )
    _append_audit(
        {
            "judge": "conclusion",
            "source_id": identity.get("source_id", ""),
            "candidate_conclusion_sha256": _sha(candidate),
            "reference_sha256": _sha(reference),
            "scope": scope,
            **details,
            "alignment_rationale": alignment_rationale,
            "decisions_rationale": decisions_rationale,
        }
    )
    return details


def _zero_conclusion_details() -> dict[str, float]:
    return {
        "alignment_score": 0.0,
        "alignment_normalized": 0.0,
        **{f"decision_{name}_score": -1.0 for name in DIMENSIONS},
        "applicable_decision_count": 0.0,
        "decisions_normalized": 0.0,
        "match_score": INVALID_REWARD,
        "judge_called": 0.0,
        "judge_latency_seconds": 0.0,
        "alignment_judge_latency_seconds": 0.0,
        "decisions_judge_latency_seconds": 0.0,
    }


# --------------------------------------------------------------------------
# Consistency judge: did the candidate's discussion develop its conclusion?
# --------------------------------------------------------------------------


def _consistency_schema(scope: dict[str, str]) -> dict:
    dimension_properties = {
        name: (
            {"type": "integer", "enum": [-1]}
            if scope[name] == "absent"
            else {"type": "integer", "minimum": 0, "maximum": 5}
        )
        for name in DIMENSIONS
    }
    return {
        "type": "object",
        "properties": {
            **dimension_properties,
            "discussion_quality": {"type": "integer", "minimum": 0, "maximum": 5},
            "discussion_conclusion_consistency": {"type": "integer", "minimum": 0, "maximum": 5},
            "grounding_and_safety": {"type": "integer", "minimum": 0, "maximum": 5},
            "major_contradiction": {"type": "boolean"},
            "unsafe_addition": {"type": "boolean"},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1600},
        },
        "required": [
            *DIMENSIONS,
            "discussion_quality",
            "discussion_conclusion_consistency",
            "grounding_and_safety",
            "major_contradiction",
            "unsafe_addition",
            "rationale",
        ],
        "additionalProperties": False,
    }


def _consistency_payload(
    candidate_discussion: str,
    candidate_conclusion: str,
    reference_discussion: str,
    reference_conclusion: str,
    scope: dict[str, str],
    evidence: str,
) -> dict:
    user_text = (
        f"<CASE_EVIDENCE>\n{evidence}\n</CASE_EVIDENCE>\n\n"
        f"<SCOPE>\n{json.dumps(scope, separators=(',', ':'))}\n</SCOPE>\n\n"
        f"<REFERENCE_DISCUSSION>\n{reference_discussion}\n</REFERENCE_DISCUSSION>\n\n"
        f"<REFERENCE_CONCLUSION>\n{reference_conclusion}\n</REFERENCE_CONCLUSION>\n\n"
        f"<CANDIDATE_DISCUSSION>\n{candidate_discussion}\n</CANDIDATE_DISCUSSION>\n\n"
        f"<CANDIDATE_CONCLUSION>\n{candidate_conclusion}\n</CANDIDATE_CONCLUSION>"
    )
    return _payload(
        CONSISTENCY_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        user_text,
        _consistency_schema(scope),
        "otb_consistency",
        int(os.environ.get("OTB_JUDGE_MAX_TOKENS", "768")),
    )


def _consistency_scores(result: dict, scope: dict[str, str]) -> dict[str, float]:
    applicable = []
    metrics: dict[str, float] = {}
    correction_count = 0
    for name in DIMENSIONS:
        raw_score = result[name]
        expected_absent = scope[name] == "absent"
        if isinstance(raw_score, bool) or not isinstance(raw_score, int) or not -1 <= raw_score <= 5:
            raise ValueError(f"invalid {name} score: {raw_score!r}")
        score = -1 if expected_absent else max(raw_score, 0)
        correction_count += int(score != raw_score)
        metrics[f"{name}_score"] = float(score)
        if not expected_absent:
            applicable.append(score / 5.0)

    for name in ("discussion_quality", "discussion_conclusion_consistency", "grounding_and_safety"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"{name} must be an integer in [0, 5], got {value!r}")

    plan = sum(applicable) / len(applicable) if applicable else 0.0
    discussion = result["discussion_quality"] / 5.0
    consistency = result["discussion_conclusion_consistency"] / 5.0
    grounding_safety = result["grounding_and_safety"] / 5.0
    match_raw = 0.45 * plan + 0.25 * discussion + 0.15 * consistency + 0.15 * grounding_safety

    consistency_factor = 0.7 if result["discussion_conclusion_consistency"] <= 1 else 1.0
    contradiction_factor = 0.6 if result["major_contradiction"] else 1.0
    unsafe_factor = 0.3 if result["unsafe_addition"] else 1.0
    multiplier = consistency_factor * contradiction_factor * unsafe_factor

    return {
        **metrics,
        "scope_score_correction_count": float(correction_count),
        "applicable_dimension_count": float(len(applicable)),
        "conclusion_plan_normalized": plan,
        "discussion_quality_score": float(result["discussion_quality"]),
        "discussion_quality_normalized": discussion,
        "discussion_conclusion_consistency_score": float(result["discussion_conclusion_consistency"]),
        "discussion_conclusion_consistency_normalized": consistency,
        "grounding_and_safety_score": float(result["grounding_and_safety"]),
        "grounding_and_safety_normalized": grounding_safety,
        "match_raw_score": match_raw,
        "reward_cap": 1.0,
        "consistency_cap_applied": float(consistency_factor < 1.0),
        "consistency_factor": consistency_factor,
        "contradiction_factor": contradiction_factor,
        "unsafe_factor": unsafe_factor,
        "safety_multiplier": multiplier,
        "match_score": match_raw * multiplier,
    }


def _support_scores(details: dict[str, float]) -> dict[str, float]:
    """The aggregate recorded in the audit log beside every consistency call."""

    plan = float(details["conclusion_plan_normalized"])
    discussion = float(details["discussion_quality_normalized"])
    consistency = float(details["discussion_conclusion_consistency_normalized"])
    grounding_safety = float(details["grounding_and_safety_normalized"])
    support = plan * consistency
    raw = 0.35 * plan + 0.20 * discussion + 0.30 * support + 0.15 * grounding_safety
    contradiction_factor = 0.6 if details["major_contradiction"] else 1.0
    unsafe_factor = 0.3 if details["unsafe_addition"] else 1.0
    safety_multiplier = contradiction_factor * unsafe_factor
    return {
        **details,
        "discussion_support_score": support,
        "discussion_support_component": 0.30 * support,
        "match_raw_score": raw,
        "reward_cap": 1.0,
        "consistency_cap_applied": 0.0,
        "consistency_factor": 1.0,
        "contradiction_factor": contradiction_factor,
        "unsafe_factor": unsafe_factor,
        "safety_multiplier": safety_multiplier,
        "match_score": raw * safety_multiplier,
    }


def judge_consistency(
    candidate_discussion: str,
    candidate_conclusion: str,
    reference_discussion: str,
    reference_conclusion: str,
    extra_info: dict | None = None,
) -> dict[str, float]:
    identity = extra_info or {}
    scope = _get_scope(reference_conclusion, identity)
    payload = _consistency_payload(
        candidate_discussion,
        candidate_conclusion,
        reference_discussion,
        reference_conclusion,
        scope,
        str(identity.get("case_evidence") or ""),
    )
    data = json.dumps(payload).encode()
    endpoint = _judge_endpoint()
    retries = int(os.environ.get("OTB_JUDGE_RETRIES", "3"))
    timeout = float(os.environ.get("OTB_JUDGE_TIMEOUT", "300"))
    error = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                endpoint + "/chat/completions", data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
            result = json.loads(raw["choices"][0]["message"]["content"])
            scores = _consistency_scores(result, scope)
            flags = {name: result[name] for name in ("major_contradiction", "unsafe_addition")}
            if not all(isinstance(value, bool) for value in flags.values()):
                raise ValueError("diagnostic flags must be booleans")
            details = {
                **scores,
                **{name: float(value) for name, value in flags.items()},
                "judge_called": 1.0,
                "judge_latency_seconds": time.perf_counter() - started,
            }
            _append_audit(
                {
                    **_support_scores(
                        {
                            "judge": "consistency",
                            "source_id": identity.get("source_id", ""),
                            "candidate_discussion_sha256": _sha(candidate_discussion),
                            "candidate_conclusion_sha256": _sha(candidate_conclusion),
                            "reference_sha256": _sha(reference_conclusion),
                            "scope": scope,
                            **details,
                            "rationale": result["rationale"],
                        }
                    )
                }
            )
            return details
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"Consistency judge failed after {retries} attempts: {error}") from error


# --------------------------------------------------------------------------
# Format score
# --------------------------------------------------------------------------

FORMAT_WEIGHTS = {
    "format_blocks": 0.2500,
    "format_envelope": 0.1875,
    "format_turn_syntax": 0.1875,
    "format_turn_numbering": 0.1875,
    "format_conclusion_sentences": 0.1250,
    "format_no_outer_whitespace": 0.0625,
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?|[一-鿿]")
REPETITION_TAIL_TOKENS = 2048
OVERLONG_START_TOKENS = 6144
OVERLONG_END_TOKENS = 8192


def _serialization_details(response: str) -> dict[str, float]:
    """Score the serialization; role composition is reported but not scored."""

    text = response or ""
    discussion = extract_discussion(text)
    conclusion = extract_conclusion(text)
    ordered = ORDERED_BLOCKS_RE.fullmatch(text)

    block_score = 0.5 * float(discussion is not None) + 0.5 * float(conclusion is not None)
    envelope_score = float(ordered is not None)
    no_outer_whitespace = float(text == text.strip() and bool(text))

    lines = discussion.splitlines() if discussion is not None else []
    matches = [TURN_RE.fullmatch(line) for line in lines]
    syntax_score = sum(match is not None for match in matches) / len(matches) if matches else 0.0
    valid_turns = [match for match in matches if match is not None]
    if valid_turns:
        numbering_score = statistics.mean(
            int(match.group("number")) == expected for expected, match in enumerate(valid_turns, 1)
        )
        roles = [match.group("role").strip() for match in valid_turns]
        role_score = float(len(set(roles)) >= 2)
        moderator_score = float(roles[0].casefold() == "moderator")
        dialogue_score = statistics.mean((role_score, moderator_score))
    else:
        numbering_score = 0.0
        roles = []
        role_score = 0.0
        moderator_score = 0.0
        dialogue_score = 0.0

    sentence_count = len(SENTENCE_END_RE.findall(conclusion.strip())) if conclusion is not None else 0
    sentence_score = 1.0 if 2 <= sentence_count <= 4 else 0.5 if sentence_count in {1, 5} else 0.0
    scored_components = {
        "format_blocks": block_score,
        "format_envelope": envelope_score,
        "format_turn_syntax": syntax_score,
        "format_turn_numbering": numbering_score,
        "format_conclusion_sentences": sentence_score,
        "format_no_outer_whitespace": no_outer_whitespace,
    }
    dense = sum(FORMAT_WEIGHTS[name] * value for name, value in scored_components.items())
    exact = float(bool(valid_turns) and all(abs(value - 1.0) < 1e-12 for value in scored_components.values()))
    return {
        "format_score": float(0.25 * dense + 0.75 * exact),
        "format_dense_score": float(dense),
        "format_exact": exact,
        "format_turn_count": float(len(valid_turns)),
        "format_role_count": float(len(set(roles))),
        "format_conclusion_sentence_count": float(sentence_count),
        "format_dialogue_roles": role_score,
        "format_moderator_first": moderator_score,
        "format_dialogue_structure": dialogue_score,
        **scored_components,
    }


def repetition_details(response: str) -> dict[str, float]:
    """Detect sustained exact phrase loops without penalizing normal reuse."""

    tokens = [token.casefold() for token in TOKEN_RE.findall(response or "")]
    tail = tokens[-REPETITION_TAIL_TOKENS:]
    max_run = 1
    if len(tail) >= 8:
        for width in range(1, 9):
            index = 0
            while index + 2 * width <= len(tail):
                block = tail[index : index + width]
                run = 1
                while tail[index + run * width : index + (run + 1) * width] == block:
                    run += 1
                max_run = max(max_run, run)
                index += max(run * width, 1)

    dominant_fraction = 0.0
    dominant_count = 0
    if len(tail) >= 4:
        counts = collections.Counter(tuple(tail[index : index + 4]) for index in range(len(tail) - 3))
        dominant_count = max(counts.values(), default=0)
        dominant_fraction = min(1.0, dominant_count * 4 / len(tail))

    # Recorded discussions do contain a few short acknowledgements in a row, so
    # eight exact repeats, or a phrase filling 15% of the tail, is the line.
    repeated = max_run >= 8 or (dominant_count >= 12 and dominant_fraction >= 0.15)
    return {
        "format_repetition_detected": float(repeated),
        "format_repetition_max_run": float(max_run),
        "format_repetition_ngram_fraction": float(dominant_fraction),
    }


def _linear_penalty(value: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("penalty end must be greater than start")
    return min(1.0, max(0.0, (float(value) - start) / (end - start)))


def _repetition_penalty(details: dict[str, float]) -> float:
    if not details["format_repetition_detected"]:
        return 0.0
    run_penalty = _linear_penalty(details["format_repetition_max_run"], 8.0, 128.0)
    ngram_penalty = _linear_penalty(details["format_repetition_ngram_fraction"], 0.15, 0.50)
    return max(run_penalty, ngram_penalty)


def _overlong_penalty(valid_response_length: int | None, response_length_limit: int | None) -> float:
    if valid_response_length is None:
        return 0.0
    end = OVERLONG_END_TOKENS
    if response_length_limit is not None:
        end = min(end, int(response_length_limit))
    if end <= OVERLONG_START_TOKENS:
        return float(int(valid_response_length) >= end)
    return _linear_penalty(valid_response_length, OVERLONG_START_TOKENS, end)


def format_details(
    response: str,
    *,
    response_truncated: bool = False,
    valid_response_length: int | None = None,
    response_length_limit: int | None = None,
) -> dict[str, float]:
    """Serialization credit, minus continuous repetition and length penalties.

    Truncation, a missing block and a severe loop fail closed instead: they set
    a negative score and a failure reason, which also skips every judge call.
    """

    base = _serialization_details(response)
    discussion_found = extract_discussion(response) is not None
    conclusion_found = extract_conclusion(response) is not None
    repetition = repetition_details(response)
    severe_repetition = bool(
        repetition["format_repetition_detected"]
        and (
            repetition["format_repetition_ngram_fraction"] >= 0.50
            or repetition["format_repetition_max_run"] >= 128
        )
    )

    penalty = 0.0
    failure_reason = 0.0
    if response_truncated:
        penalty = -1.0
        failure_reason = 1.0
    elif severe_repetition:
        penalty = -1.0
        failure_reason = 2.0
    elif not discussion_found and not conclusion_found:
        penalty = -1.0
        failure_reason = 3.0
    elif not discussion_found or not conclusion_found:
        penalty = -0.75
        failure_reason = 4.0

    repetition_penalty = _repetition_penalty({**base, **repetition})
    overlong_penalty = _overlong_penalty(valid_response_length, response_length_limit)
    if failure_reason:
        format_score = penalty
    else:
        format_score = max(-1.0, base["format_score"] - repetition_penalty - overlong_penalty)

    return {
        **base,
        **repetition,
        "format_base_score": float(base["format_score"]),
        "format_score": float(format_score),
        "format_failure_penalty": float(penalty),
        # 0=none, 1=truncated, 2=repetition, 3=no blocks, 4=one block.
        "format_failure_reason": failure_reason,
        "format_complete": float(discussion_found and conclusion_found),
        "response_truncated": float(response_truncated),
        "format_severe_repetition": float(severe_repetition),
        "format_repetition_penalty": float(repetition_penalty),
        "format_overlong_penalty": float(overlong_penalty),
        "valid_response_length": float(valid_response_length or 0),
        "response_length_limit": float(response_length_limit or 0),
    }


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


def _zero_details() -> dict[str, float]:
    return {
        **_zero_conclusion_details(),
        "discussion_conclusion_consistency_normalized": 0.0,
        "consistency_judge_latency_seconds": 0.0,
        "discussion_support_score": 0.0,
        "match_score": INVALID_REWARD,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, kwargs
    identity = extra_info or {}
    structure = format_details(
        solution_str,
        response_truncated=bool(identity.get("response_truncated", False)),
        valid_response_length=identity.get("valid_response_length"),
        response_length_limit=identity.get("response_length_limit"),
    )
    candidate_discussion = extract_discussion(solution_str)
    candidate_conclusion = extract_conclusion(solution_str)
    reference_discussion, reference_conclusion = extract_reference_parts(ground_truth)
    if (
        structure["format_failure_reason"]
        or candidate_discussion is None
        or candidate_conclusion is None
        or structure["format_severe_repetition"]
    ):
        details = _zero_details()
    else:
        conclusion = judge_conclusion(candidate_conclusion, reference_conclusion, identity)
        started = time.perf_counter()
        integrated = judge_consistency(
            candidate_discussion,
            candidate_conclusion,
            reference_discussion,
            reference_conclusion,
            identity,
        )
        consistency = float(integrated["discussion_conclusion_consistency_normalized"])
        if not 0.0 <= consistency <= 1.0:
            raise ValueError(f"invalid consistency: {consistency!r}")
        consistency_latency = time.perf_counter() - started
        alignment = conclusion["alignment_normalized"]
        decisions = conclusion["decisions_normalized"]
        support = decisions * consistency
        match = (
            ALIGNMENT_WEIGHT * alignment
            + SUPPORT_WEIGHT * support
            + DECISIONS_WEIGHT * decisions
        )
        details = {
            **conclusion,
            "discussion_conclusion_consistency_normalized": consistency,
            "consistency_judge_latency_seconds": consistency_latency,
            "discussion_support_score": support,
            "match_score": match,
        }
        _append_audit(
            {
                "judge": "reward",
                "source_id": identity.get("source_id", ""),
                "consistency": consistency,
                "support": support,
                "match_score": match,
            }
        )
    return {
        "score": details["match_score"] + FORMAT_WEIGHT * structure["format_score"],
        **structure,
        "discussion_found": float(candidate_discussion is not None),
        "conclusion_found": float(candidate_conclusion is not None),
        **details,
    }
