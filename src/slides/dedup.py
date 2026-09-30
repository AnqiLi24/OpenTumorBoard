from __future__ import annotations

import logging
from typing import List

import numpy as np

log = logging.getLogger(__name__)


def deduplicate(
    embeddings: np.ndarray,
    sim_threshold: float = 0.92,
) -> List[int]:
    """
    Greedy cosine-similarity deduplication.
    Keep frame i if cosine-sim to every already-kept frame < sim_threshold.
    Returns indices of kept frames in original order.

    Lower threshold → keep more frames (more granular transitions).
    Higher threshold → keep fewer (more aggressive dedup).
    """
    kept: List[int] = []
    kept_embs: List[np.ndarray] = []

    for i, emb in enumerate(embeddings):
        if not kept_embs:
            kept.append(i)
            kept_embs.append(emb)
            continue
        stack = np.stack(kept_embs)
        sims = stack @ emb
        if sims.max() < sim_threshold:
            kept.append(i)
            kept_embs.append(emb)

    return kept
