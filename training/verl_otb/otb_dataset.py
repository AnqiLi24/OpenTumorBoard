"""OpenTumorBoard dataset adapter for literal ``<image>`` tokens in the system prompt."""

import os

from verl.utils.dataset.rl_dataset import RLHFDataset


_SENTINEL = "\x00OTB_LITERAL_IMAGE_TOKEN\x00"


def remap_nfs_path(path: str) -> str:
    """Translate an absolute NFS prefix when the same export is mounted elsewhere.

    The parquet stays immutable; a job on a host that mounts the image directory
    under another prefix opts into translation with ``OTB_NFS_PATH_FROM`` and
    ``OTB_NFS_PATH_TO``.
    """

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


class OTBTumorBoardDataset(RLHFDataset):
    """Treat the three example ``<image>`` strings in the system turn as prose.

    The user turn's ``<image>`` strings remain real placeholders and are consumed by
    :class:`RLHFDataset` in order. The original prompt text is restored byte-for-byte.
    """

    def _build_messages(self, example: dict, key: str):
        images = example.get(self.image_key, None)
        if images is not None:
            example[self.image_key] = [
                remap_nfs_path(os.fspath(image)) if isinstance(image, str | os.PathLike) else image
                for image in images
            ]
        messages = example[key]
        hidden = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "system" and isinstance(content, str) and "<image>" in content:
                hidden.append((message, content))
                message["content"] = content.replace("<image>", _SENTINEL)

        # Explicit parent dispatch is required when verl imports custom classes under a
        # synthetic module name in Ray dataset-filter workers.
        output = RLHFDataset._build_messages(self, example, key)

        for message, original in hidden:
            content = message["content"]
            if isinstance(content, str):
                message["content"] = content.replace(_SENTINEL, "<image>")
            else:
                for part in content:
                    if part.get("type") == "text" and _SENTINEL in part.get("text", ""):
                        part["text"] = part["text"].replace(_SENTINEL, "<image>")
                if any(part.get("type") != "text" for part in content):
                    raise AssertionError("system turn produced a non-text multimodal part")
                if "".join(part["text"] for part in content) != original:
                    raise AssertionError("system prompt changed while shielding literal <image> tokens")
            if _SENTINEL in str(message["content"]):
                raise AssertionError("literal-image sentinel leaked into model input")

        return output

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        """Keep the multimodal helper available in HF Dataset worker processes.

        ``datasets.Dataset.filter(num_proc=...)`` reconstructs a dynamically loaded
        custom class in its worker processes.  In this verl/Python combination,
        inherited classmethods are not retained on that reconstructed object even
        though methods declared directly on the custom class are.  Dispatching to
        the parent explicitly avoids filtering every valid sample as an error.
        """
        return RLHFDataset._process_multi_modal_info(
            messages,
            image_patch_size=image_patch_size,
            config=config,
        )

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        """Apply the same explicit dispatch in rollout agent processes."""
        return await RLHFDataset.process_multi_modal_info(
            messages,
            image_patch_size=image_patch_size,
            config=config,
        )
