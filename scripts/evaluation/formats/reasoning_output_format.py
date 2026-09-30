#!/usr/bin/env python3
"""Deterministically separate a visible reasoning trace from its final response."""

from __future__ import annotations

import re


FINAL_HEADING = re.compile(
    r"(?im)^\s*#{1,6}\s*(?:\*\*)?final\s+(?:answer|response)"
    r"(?:\*\*)?\s*:?\s*(?P<inline>.*)$"
)


def extract_final_response(
    content: str, reasoning_content: str | None = None
) -> tuple[str | None, str]:
    """Extract only a model's explicitly separated final response.

    The complete provider fields remain stored by the runners. This helper is
    intentionally conservative: it accepts provider-separated reasoning,
    Markdown Final Answer/Response headings, or a closed ``<think>`` prefix.
    It never guesses a final answer from an unfinished reasoning trace.
    """
    if reasoning_content is not None:
        final = content.strip()
        if final:
            return final, "provider_separated_reasoning"
        return None, "empty_provider_final_response"

    matches = list(FINAL_HEADING.finditer(content))
    if matches:
        match = matches[-1]
        inline = match.group("inline").strip()
        suffix = content[match.end() :].strip()
        final = "\n".join(part for part in (inline, suffix) if part).strip()
        if not final:
            return None, "empty_markdown_final_response"
        heading = match.group(0).casefold()
        label = "final_answer" if "final answer" in heading else "final_response"
        return final, f"markdown_{label}"

    closing = list(re.finditer(r"</think\s*>", content, re.IGNORECASE))
    if closing:
        final = content[closing[-1].end() :].strip()
        if final:
            return final, "closed_think_suffix"
        return None, "empty_think_suffix"

    bracket_closing = list(re.finditer(r"\[/think\]", content, re.IGNORECASE))
    if bracket_closing:
        final = content[bracket_closing[-1].end() :].strip()
        if final:
            return final, "closed_bracket_think_suffix"
        return None, "empty_bracket_think_suffix"

    return None, "missing_explicit_final_boundary"


def extract_markdown_final_response(content: str) -> tuple[str | None, str]:
    """Backward-compatible alias used by earlier local tests."""
    return extract_final_response(content)
