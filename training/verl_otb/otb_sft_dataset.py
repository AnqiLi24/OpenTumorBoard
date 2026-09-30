"""OpenTumorBoard adapter for verl's multimodal SFT dataset."""

from __future__ import annotations

import os
from copy import deepcopy

import torch
from omegaconf import OmegaConf

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template

_SENTINEL = "\x00OTB_SFT_LITERAL_IMAGE_TOKEN\x00"


def remap_nfs_path(path: str) -> str:
    """Translate an absolute path prefix when the image directory is mounted elsewhere."""

    source = os.environ.get("OTB_NFS_PATH_FROM")
    target = os.environ.get("OTB_NFS_PATH_TO")
    if not source and not target:
        return path
    if not source or not target:
        raise ValueError("OTB_NFS_PATH_FROM and OTB_NFS_PATH_TO must be set together")
    source = source.rstrip("/")
    target = target.rstrip("/")
    if path == source:
        return target
    if path.startswith(source + "/"):
        return target + path[len(source) :]
    return path


class OTBTumorBoardSFTDataset(MultiTurnSFTDataset):
    """Expand images in the case turn while preserving example markers as prose."""

    def _apply_kwargs(self, enable_thinking=None):
        """Return ordinary containers and keep all processor-only args nested."""

        raw_kwargs = self.apply_chat_template_kwargs
        if OmegaConf.is_config(raw_kwargs):
            apply_kwargs = OmegaConf.to_container(raw_kwargs, resolve=True)
        else:
            apply_kwargs = deepcopy(dict(raw_kwargs))
        if not isinstance(apply_kwargs, dict):
            raise TypeError("apply_chat_template_kwargs must resolve to a mapping")

        processor_kwargs = apply_kwargs.pop("processor_kwargs", {})
        if not isinstance(processor_kwargs, dict):
            raise TypeError("processor_kwargs must resolve to a mapping")
        processor_kwargs["return_tensors"] = "pt"
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = enable_thinking
        return apply_kwargs, processor_kwargs

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        """Keep image limits when Transformers also needs tensor-return kwargs.

        Transformers 5.5 replaces an explicit ``processor_kwargs`` mapping when
        any processor kwarg is also supplied at the top level.  verl's generic
        SFT dataset supplies ``return_tensors`` at the top level, which silently
        discards OTB's image-size limit and can expand a sample beyond 190k
        tokens. Put every processor kwarg in the nested mapping to avoid that
        overwrite.
        """

        processor = self.processor if self.processor is not None else self.tokenizer
        apply_kwargs, processor_kwargs = self._apply_kwargs(enable_thinking)

        inputs = apply_chat_template(
            processor,
            messages=[message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            processor_kwargs=processor_kwargs,
            **apply_kwargs,
        )
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)
        return input_ids, loss_mask, attention_mask, inputs

    def sanity_check(self, input_ids, messages, tools, enable_thinking):
        """Render the full example with the exact same processor settings."""

        processor = self.processor if self.processor is not None else self.tokenizer
        apply_kwargs, processor_kwargs = self._apply_kwargs(enable_thinking)
        inputs = apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            processor_kwargs=processor_kwargs,
            **apply_kwargs,
        )
        expected = inputs["input_ids"].squeeze(0)
        if torch.equal(input_ids, expected):
            return

        error_message = (
            "Per-turn OTB tokenization does not equal full-conversation tokenization. "
            "Set data.ignore_input_ids_mismatch=True only after inspecting the mismatch."
        )
        if self.ignore_input_ids_mismatch:
            import logging

            logging.getLogger(__name__).warning(error_message)
        else:
            raise AssertionError(error_message)

    def _build_messages(self, example: dict):
        example = dict(example)
        messages = convert_nested_value_to_list_recursive(example[self.messages_key])
        images = example.get(self.image_key)
        if images is not None:
            normalized_images = []
            for image in images:
                if isinstance(image, str | os.PathLike):
                    path = remap_nfs_path(os.fspath(image))
                    normalized_images.append({"image": path if path.startswith("file://") else "file://" + path})
                else:
                    normalized_images.append(image)
            example[self.image_key] = normalized_images

        expected_system_text = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "system" and isinstance(content, str):
                expected_system_text.append(content)
                if "<image>" in content:
                    message["content"] = content.replace("<image>", _SENTINEL)
        example[self.messages_key] = messages

        output = MultiTurnSFTDataset._build_messages(self, example)

        restored = []
        for message in output:
            content = message.get("content")
            if message.get("role") != "system":
                continue
            if isinstance(content, str):
                message["content"] = content.replace(_SENTINEL, "<image>")
                restored.append(message["content"])
                continue
            text = []
            for part in content:
                if part.get("type") != "text":
                    raise AssertionError("system turn produced a non-text multimodal part")
                part["text"] = part.get("text", "").replace(_SENTINEL, "<image>")
                text.append(part["text"])
            restored.append("".join(text))

        if restored != expected_system_text:
            raise AssertionError("system prompt changed while shielding literal <image> tokens")
        if _SENTINEL in str(output):
            raise AssertionError("literal-image sentinel leaked into SFT model input")
        return output
