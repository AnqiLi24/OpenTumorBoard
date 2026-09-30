"""
slides — robust slide extraction from conference-room lecture videos.

Stages
------
1  sampling    sample_frames()
2  detectors   SlideDetector hierarchy + build_detector()
3  crop        crop_slide()
4  embedder    CLIPEmbedder
5  dedup       deduplicate()
6  gpt_verify  GPTSlideVerifier
7  pipeline    run_pipeline()  (orchestrates all stages)
"""

from .crop import crop_slide
from .dedup import deduplicate
from .detectors import (
    BBox,
    CVSlideDetector,
    GroundedSAM2SlideDetector,
    SAM2SlideDetector,
    SAMSlideDetector,
    SlideDetector,
    build_detector,
)
from .embedder import CLIPEmbedder
from .gpt_verify import GPTSlideVerifier
from .pipeline import run_pipeline
from .sampling import sample_frames

__all__ = [
    "sample_frames",
    "BBox",
    "SlideDetector",
    "CVSlideDetector",
    "SAMSlideDetector",
    "SAM2SlideDetector",
    "GroundedSAM2SlideDetector",
    "build_detector",
    "crop_slide",
    "CLIPEmbedder",
    "deduplicate",
    "GPTSlideVerifier",
    "run_pipeline",
]
