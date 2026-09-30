from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    Compute L2-normalised CLIP image embeddings via HuggingFace transformers.
    Default model: openai/clip-vit-base-patch32  (~600 MB, fast)
    Larger model:  openai/clip-vit-large-patch14 (better quality)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        log.info(f"CLIP loaded: {model_name}")

    @torch.no_grad()
    def embed(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """Return L2-normalised image embeddings, shape (N, D)."""
        pil_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in images_bgr
        ]
        inputs = self.processor(images=pil_imgs, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised text embeddings, shape (N, D)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def slide_scores(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Cosine similarity of each frame to the concept 'presentation slide'.
        Returns shape (N,) in [-1, 1]; higher = more likely a real slide.
        Score = sim(frame, pos_text) - sim(frame, neg_text).
        """
        pos = "a presentation slide projected on a screen"
        neg = "the speaker, an audience or a conference room without a screen"
        text_embs = self.embed_texts([pos, neg])   # (2, D)
        img_embs = self.embed(images_bgr)           # (N, D)
        sims = img_embs @ text_embs.T              # (N, 2)
        return sims[:, 0] - sims[:, 1]
