from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import torch

from .detectors import build_detector
from .embedder import CLIPEmbedder
from .gpt_verify import GPTSlideVerifier
from .pipeline import run_pipeline

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="slides: slide extraction with SAM2 + CLIP deduplication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument(
        "--output-dir", "--output",
        dest="output_dir",
        default="./processed",
        help="Root output directory",
    )
    parser.add_argument(
        "--case-seg",
        default=None,
        help="Path to case_segmentation.json (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Path to transcript.with_speakers.json for caption generation",
    )
    parser.add_argument(
        "--caption-model",
        default=None,
        help="Azure OpenAI deployment name for caption generation (defaults to DEPLOYMENT in gpt_client)",
    )

    # Sampling
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=0.5,
        help="Frames per second to sample (0.5 = one frame every 2 s)",
    )

    # Detector
    parser.add_argument(
        "--detector",
        choices=["cv", "sam", "sam2", "gsam2"],
        default="cv",
        help="Slide-region detector backend",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to SAM1 / SAM2 / SAM2.1 model checkpoint (.pth / .pt)",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        help="SAM1 model type: vit_h | vit_l | vit_b",
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path (relative to sam2 package root)",
    )
    parser.add_argument(
        "--gdino-config",
        default=None,
        help="[gsam2] Path to GroundingDINO config .py file",
    )
    parser.add_argument(
        "--gdino-checkpoint",
        default=None,
        help="[gsam2] Path to GroundingDINO checkpoint .pth file",
    )
    parser.add_argument(
        "--gdino-prompt",
        default="projection screen . presentation slide . powerpoint slide",
        help="[gsam2] Grounding DINO text prompt (dot-separated noun phrases)",
    )
    parser.add_argument(
        "--gdino-box-threshold",
        type=float,
        default=0.30,
        help="[gsam2] Grounding DINO box confidence threshold",
    )
    parser.add_argument(
        "--min-slide-area",
        type=float,
        default=0.10,
        help="Minimum slide area as a fraction of the full frame [0–1]. "
             "Detected regions smaller than this are discarded (default 0.10 = 10%%).",
    )

    # CLIP / dedup
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.92,
        help="Cosine-similarity threshold for dedup [0, 1]; higher = more aggressive",
    )
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument(
        "--detect-batch-size",
        type=int,
        default=8,
        help="Frames per detection batch (default 8)",
    )
    parser.add_argument(
        "--clip-verify-threshold",
        type=float,
        default=None,
        help=(
            "CLIP pre-filter threshold applied to FULL FRAMES before detection. "
            "Score = sim(frame, 'presentation slide') − sim(frame, 'audience/room'). "
            "Omit to disable. "
            "Full-frame scores are lower than crop scores (slide is a small part of the frame), "
            "so use a negative value like -0.02 to be permissive, "
            "or 0.0 to be strict."
        ),
    )

    # GPT verification
    parser.add_argument(
        "--gpt-verify",
        action="store_true",
        default=False,
        help="Run a final GPT-5.4 vision pass to keep only clinically useful slides",
    )
    parser.add_argument(
        "--gpt-batch-size",
        type=int,
        default=5,
        help="Number of slide images per GPT API call (default 5)",
    )

    # Misc
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--keep-undetected",
        action="store_true",
        help="Use full frame for frames where no slide is detected",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save annotated detection frames to output-dir/debug_frames/",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    case_seg = args.case_seg
    if case_seg is None:
        candidate = os.path.join(output_dir, "case_segmentation.json")
        if os.path.exists(candidate):
            case_seg = candidate
            log.info(f"Auto-detected case segmentation: {case_seg}")

    transcript = args.transcript
    if transcript is None:
        candidate = os.path.join(output_dir, "transcript.with_speakers.json")
        if os.path.exists(candidate):
            transcript = candidate
            log.info(f"Auto-detected transcript: {transcript}")

    log.info(f"Building detector: {args.detector}")
    detector = build_detector(
        args.detector,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam2_config=args.sam2_config,
        device=args.device,
        min_area_ratio=args.min_slide_area,
        gdino_config=args.gdino_config,
        gdino_checkpoint=args.gdino_checkpoint,
        gdino_text_prompt=args.gdino_prompt,
        gdino_box_threshold=args.gdino_box_threshold,
    )

    log.info(f"Loading CLIP: {args.clip_model}")
    embedder = CLIPEmbedder(args.clip_model, args.device)

    gpt_verifier: Optional[GPTSlideVerifier] = None
    if args.gpt_verify:
        log.info("GPT verification enabled")
        gpt_verifier = GPTSlideVerifier(batch_size=args.gpt_batch_size)

    from .detectors import CVSlideDetector
    fallback_detector = None
    if args.detector != "cv":
        log.info("CV fallback detector enabled (triggers if primary finds 0 crops per case)")
        fallback_detector = CVSlideDetector(min_area_ratio=args.min_slide_area)

    from gpt_client import DEPLOYMENT
    run_pipeline(
        video_path=args.video,
        output_dir=output_dir,
        case_seg_path=case_seg,
        detector=detector,
        embedder=embedder,
        sample_fps=args.sample_fps,
        sim_threshold=args.sim_threshold,
        clip_verify_threshold=args.clip_verify_threshold,
        keep_undetected=args.keep_undetected,
        embed_batch_size=args.embed_batch_size,
        detect_batch_size=args.detect_batch_size,
        debug=args.debug,
        gpt_verifier=gpt_verifier,
        transcript_path=transcript,
        caption_model=args.caption_model or DEPLOYMENT,
        fallback_detector=fallback_detector,
    )
