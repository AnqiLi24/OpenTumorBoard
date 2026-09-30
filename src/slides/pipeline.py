from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from .crop import crop_slide
from .dedup import deduplicate
from .detectors import SlideDetector
from .embedder import CLIPEmbedder
from .gpt_verify import GPTSlideVerifier
from .sampling import sample_frames
from gpt_client import client, DEPLOYMENT

log = logging.getLogger(__name__)

_CAPTION_CONTEXT_BEFORE_SEC = 60


def _get_utterances_for_slide(utterances: list, start_sec: float, end_sec: float) -> list:
    context_start = max(0.0, start_sec - _CAPTION_CONTEXT_BEFORE_SEC)
    return [
        u for u in utterances
        if u.get("end_sec", 0.0) >= context_start and u.get("start_sec", 0.0) <= end_sec
    ]


def _img_to_b64(img_bgr: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _generate_caption(
    utterances: list,
    slide_start: float,
    slide_end: float,
    model: str,
    image: Optional[np.ndarray] = None,
) -> str:
    lines = [
        f"[{u.get('start_sec', 0):.1f}s-{u.get('end_sec', 0):.1f}s] "
        f"{u.get('speaker_id', 'Speaker')}: {u.get('text', '').strip()}"
        for u in utterances
    ]
    transcript_block = "\n".join(lines) if lines else "(no transcript context available)"
    prompt_text = (
        f"The following transcript excerpt is from a medical tumor board discussion. "
        f"A slide is displayed from {slide_start:.1f}s to {slide_end:.1f}s.\n\n"
        f"Transcript context:\n{transcript_block}\n\n"
        "The slide image is attached above. "
        "Write a brief and concise caption (1 sentence) describing this image as it appears, "
        "taking into account the slide content and the discussion context. "
        "The caption should describe the subject or question being explored, "
        "but must NOT reveal specific conclusions, diagnoses, or treatment decisions. "
        "Be concise and objective."
    )

    content: list = []
    if image is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64(image)}"},
        })
    content.append({"type": "text", "text": prompt_text})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a medical conference assistant helping label slides with neutral, descriptive captions."},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _detect_and_crop(
    raw_frames: list,
    detector: SlideDetector,
    detect_batch_size: int,
    keep_undetected: bool,
    debug: bool,
    debug_dir: Optional[str],
    case_id: str,
) -> tuple:
    """Run detector over raw_frames; return (timestamps, crops, n_undetected)."""
    timestamps: List[float] = []
    crops: List[np.ndarray] = []
    n_undetected = 0

    for batch_start in range(0, len(raw_frames), detect_batch_size):
        batch = raw_frames[batch_start : batch_start + detect_batch_size]
        ts_batch = [ts for ts, _ in batch]
        frames_batch = [f for _, f in batch]

        bboxes = detector.detect_batch(frames_batch)

        for ts, frame, bbox in zip(ts_batch, frames_batch, bboxes):
            if bbox is None:
                n_undetected += 1
                if not keep_undetected:
                    continue
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)

            cropped = crop_slide(frame, bbox)
            if cropped.size == 0:
                continue

            timestamps.append(ts)
            crops.append(cropped)

            if debug and debug_dir:
                x, y, bw, bh = bbox
                vis = frame.copy()
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                cv2.imwrite(os.path.join(debug_dir, f"{case_id}_{ts:.1f}.jpg"), vis)

    return timestamps, crops, n_undetected


def run_pipeline(
    video_path: str,
    output_dir: str,
    case_seg_path: Optional[str],
    detector: SlideDetector,
    embedder: CLIPEmbedder,
    sample_fps: float = 0.5,
    sim_threshold: float = 0.92,
    clip_verify_threshold: Optional[float] = None,
    keep_undetected: bool = False,
    embed_batch_size: int = 16,
    detect_batch_size: int = 8,
    debug: bool = False,
    gpt_verifier: Optional[GPTSlideVerifier] = None,
    transcript_path: Optional[str] = None,
    caption_model: str = DEPLOYMENT,
    fallback_detector: Optional[SlideDetector] = None,
) -> None:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if debug:
        debug_dir = os.path.join(output_dir, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)

    # ── load cases ──────────────────────────────────────────────────────────
    if case_seg_path and os.path.exists(case_seg_path):
        with open(case_seg_path, encoding="utf-8") as f:
            cases = json.load(f).get("cases", [])
        log.info(f"Loaded {len(cases)} cases from {case_seg_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        duration = total / fps
        cases = [{"case_id": "case_001", "start_sec": 0.0, "end_sec": duration}]
        log.warning("No case segmentation — treating entire video as case_001")

    all_slides: List[dict] = []
    slide_idx = 1

    for case in cases:
        case_id: str = case["case_id"]
        start_sec: float = case["start_sec"]
        end_sec: float = case["end_sec"]
        if end_sec <= start_sec:
            continue

        log.info(f"Case {case_id}: {start_sec:.1f}s – {end_sec:.1f}s")

        # Stage 1 — sample
        raw_frames = sample_frames(video_path, start_sec, end_sec, sample_fps)
        if not raw_frames:
            continue

        # Stage 2 — CLIP pre-filter on full frames (batched, fast)
        # Runs before the expensive detector so we only detect on likely-slide frames.
        if clip_verify_threshold is not None:
            all_scores: List[np.ndarray] = []
            for i in range(0, len(raw_frames), embed_batch_size):
                batch = [f for _, f in raw_frames[i : i + embed_batch_size]]
                all_scores.append(embedder.slide_scores(batch))
            scores = np.concatenate(all_scores)
            n_before = len(raw_frames)
            raw_frames = [
                (ts, f) for (ts, f), s in zip(raw_frames, scores)
                if s >= clip_verify_threshold
            ]
            log.info(
                f"  CLIP pre-filter (threshold={clip_verify_threshold:+.2f}): "
                f"{len(raw_frames)}/{n_before} frames kept for detection"
            )
            if not raw_frames:
                continue

        # Stage 3 — detect & crop (batched)
        timestamps, crops, n_undetected = _detect_and_crop(
            raw_frames, detector, detect_batch_size,
            keep_undetected, debug, debug_dir if debug else None, case_id,
        )
        log.info(
            f"  {len(crops)} crops after detection "
            f"({n_undetected} undetected, {'kept' if keep_undetected else 'skipped'})"
        )

        if not crops and fallback_detector is not None:
            log.info(f"  Primary detector found 0 crops → retrying with CV fallback")
            timestamps, crops, n_undetected = _detect_and_crop(
                raw_frames, fallback_detector, detect_batch_size,
                keep_undetected, debug, debug_dir if debug else None, case_id,
            )
            log.info(f"  {len(crops)} crops after CV fallback detection")

        if not crops:
            continue

        # Stage 4 — CLIP embeddings on crops (for dedup)
        all_embs: List[np.ndarray] = []
        for i in range(0, len(crops), embed_batch_size):
            all_embs.append(embedder.embed(crops[i : i + embed_batch_size]))
        embeddings = np.concatenate(all_embs, axis=0)

        # Stage 5 — deduplication
        kept = deduplicate(embeddings, sim_threshold)
        log.info(f"  {len(kept)} unique slides after dedup (from {len(crops)})")

        # Stage 6 — GPT vision verification (optional)
        if gpt_verifier is not None and kept:
            kept_crops = [crops[i] for i in kept]
            verdicts = gpt_verifier.verify(kept_crops)
            kept = [idx for idx, ok in zip(kept, verdicts) if ok]
            log.info(f"  {len(kept)} slides kept after GPT verification")

        # Stage 7 — assign time ranges and write files
        for rank, idx in enumerate(kept):
            slide_id = f"slide_{slide_idx:03d}"
            slide_idx += 1

            if rank + 1 < len(kept):
                slide_end = timestamps[kept[rank + 1]]
            else:
                slide_end = end_sec

            frame_name = f"{slide_id}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_name), crops[idx])

            all_slides.append({
                "slide_id": slide_id,
                "case_id": case_id,
                "start_sec": timestamps[idx],
                "end_sec": slide_end,
                "frame_path": f"frames/{frame_name}",
                "caption": "",
            })

    # Generate captions
    transcript_utterances: List[dict] = []
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf-8") as f:
            transcript_utterances = json.load(f).get("utterances", [])
        log.info(f"Loaded {len(transcript_utterances)} utterances for caption generation")
    elif transcript_path:
        log.warning(f"Transcript not found: {transcript_path}; captions will be empty")

    if transcript_utterances:
        log.info(f"Generating captions for {len(all_slides)} slides...")
        for slide in all_slides:
            relevant = _get_utterances_for_slide(
                transcript_utterances, slide["start_sec"], slide["end_sec"]
            )
            img = cv2.imread(os.path.join(output_dir, slide["frame_path"]))
            try:
                slide["caption"] = _generate_caption(
                    relevant, slide["start_sec"], slide["end_sec"], caption_model, image=img
                )
            except Exception as e:
                log.warning(f"Caption generation failed for {slide['slide_id']}: {e}")
            log.info(f"  {slide['slide_id']}: {slide['caption'][:80]}")

    # Write JSON
    output = {"video_id": video_id, "slides": all_slides}
    json_path = os.path.join(output_dir, "slides.raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Done: {len(all_slides)} unique slides → {output_dir}")
    log.info(f"JSON: {json_path}")
