from __future__ import annotations

import json
import logging
from typing import List

import cv2
import numpy as np

log = logging.getLogger(__name__)

_GPT_VERIFY_PROMPT = """\
You are reviewing frames extracted from a medical tumor board presentation video.
Each image below is a candidate slide crop. For each one, decide:
  YES – it is a readable presentation slide that contains clinically useful information
        (patient case details, genomic data, pathology results, treatment plans,
         drug recommendations, imaging findings, or structured medical content).
  NO  – it is NOT a useful slide: blurry, a photo of the room/audience/speaker,
        a blank/title-only screen, a non-clinical decorative image, or unreadable.

Respond ONLY with a valid JSON array of booleans (true/false), one per image, in order.
Example for 3 images: [true, false, true]
Do NOT include any explanation."""


class GPTSlideVerifier:
    """
    Final verification pass: send slide crops to GPT-5.4 vision, keep only
    those the model judges to contain useful clinical content.

    Images are packed N-per-request to reduce API calls and latency.
    Requires gpt_client.py to be importable (lives in src/ alongside this package).
    """

    def __init__(self, batch_size: int = 5):
        from gpt_client import client, DEPLOYMENT  # src/ inserted into sys.path by __main__
        self.client = client
        self.deployment = DEPLOYMENT
        self.batch_size = batch_size
        log.info(f"GPT verifier ready (deployment={DEPLOYMENT}, batch_size={batch_size})")

    @staticmethod
    def _to_b64(img_bgr: np.ndarray) -> str:
        import base64
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()

    def verify(self, crops: List[np.ndarray]) -> List[bool]:
        """Return a bool list (same length as crops): True = keep."""
        results: List[bool] = []
        for i in range(0, len(crops), self.batch_size):
            results.extend(self._call_gpt(crops[i : i + self.batch_size]))
        return results

    def _call_gpt(self, batch: List[np.ndarray]) -> List[bool]:
        content: List[dict] = [{"type": "text", "text": _GPT_VERIFY_PROMPT}]
        for idx, img in enumerate(batch, 1):
            content.append({"type": "text", "text": f"Image {idx}:"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._to_b64(img)}",
                },
            })

        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            verdicts = json.loads(raw)
            if isinstance(verdicts, list) and len(verdicts) == len(batch):
                return [bool(v) for v in verdicts]
            log.warning(f"  GPT returned unexpected response: {raw!r}; keeping all")
            return [True] * len(batch)
        except Exception as e:
            log.warning(f"  GPT verify failed ({e}); keeping all in batch")
            return [True] * len(batch)
