from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def sample_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 0.5,
) -> List[Tuple[float, np.ndarray]]:
    """Uniformly sample frames at `sample_fps` frames/sec from [start_sec, end_sec].
    Returns list of (timestamp_sec, frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    interval = 1.0 / max(sample_fps, 1e-6)
    frames: List[Tuple[float, np.ndarray]] = []
    t = start_sec
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if ret:
            frames.append((round(t, 3), frame))
        t += interval
    cap.release()
    log.info(f"  sampled {len(frames)} frames from [{start_sec:.1f}s, {end_sec:.1f}s]")
    return frames
