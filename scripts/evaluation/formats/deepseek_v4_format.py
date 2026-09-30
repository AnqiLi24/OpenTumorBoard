#!/usr/bin/env python3
"""Minimal, auditable DeepSeek-V4 chat encoding for two-turn evaluation prompts.

The benchmark manifests contain exactly one system turn followed by one user
turn. This reproduces ``encode_messages(..., thinking_mode="chat")`` from the
official DeepSeek-V4 checkpoint without copying its unrelated tool machinery.
"""

from __future__ import annotations

from typing import Any


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINKING_END_TOKEN = "</think>"
THINKING_START_TOKEN = "<think>"


def encode_deepseek_v4_chat(messages: list[dict[str, Any]]) -> str:
    """Encode one system + one user turn in official non-thinking chat mode."""
    if len(messages) != 2:
        raise ValueError("DeepSeek-V4 benchmark prompts require exactly two turns")
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("DeepSeek-V4 benchmark prompts require system then user")
    system = messages[0].get("content")
    user = messages[1].get("content")
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError("DeepSeek-V4 message content must be text")
    return (
        f"{BOS_TOKEN}{system}{USER_TOKEN}{user}"
        f"{ASSISTANT_TOKEN}{THINKING_END_TOKEN}"
    )


def encode_deepseek_v4_thinking(messages: list[dict[str, Any]]) -> str:
    """Encode one system + one user turn in official thinking mode."""
    if len(messages) != 2:
        raise ValueError("DeepSeek-V4 benchmark prompts require exactly two turns")
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("DeepSeek-V4 benchmark prompts require system then user")
    system = messages[0].get("content")
    user = messages[1].get("content")
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError("DeepSeek-V4 message content must be text")
    return (
        f"{BOS_TOKEN}{system}{USER_TOKEN}{user}"
        f"{ASSISTANT_TOKEN}{THINKING_START_TOKEN}"
    )
