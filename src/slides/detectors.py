from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]   # x, y, w, h  (top-left origin)


# ─── Abstract base ────────────────────────────────────────────────────────────

class SlideDetector(ABC):
    """Abstract detector: given a BGR frame, return a BBox or None."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        ...

    def detect_batch(self, frames_bgr: List[np.ndarray]) -> List[Optional[BBox]]:
        """Batch detection. Default: sequential loop. Override for GPU batching."""
        return [self.detect(f) for f in frames_bgr]


# ─── 2a. Classical CV (always available, no ML) ───────────────────────────────

class CVSlideDetector(SlideDetector):
    """
    Heuristic projection-screen detector.

    Conference-room screens are usually:
      • Bright (high luminance compared to audience/room)
      • Large  (>10 % of frame area)
      • Landscape-oriented (aspect 1.1 – 2.6)
      • In the upper-center portion of the frame

    Strategy
    --------
    1. Isolate bright pixels (top-k percentile threshold).
    2. Morphological close to merge nearby bright blobs.
    3. Find external contours; score each by area × rectangularity × center-bias.
    4. Return bbox of highest-scoring candidate.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.08,
        max_area_ratio: float = 0.75,
        aspect_min: float = 1.1,
        aspect_max: float = 2.6,
        brightness_pct: float = 70.0,
        min_std: float = 12.0,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.brightness_pct = brightness_pct
        self.min_std = min_std

    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        h, w = frame_bgr.shape[:2]
        img_area = h * w

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        threshold = float(np.percentile(blur, self.brightness_pct))
        _, bright = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[BBox] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * self.min_area_ratio:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = bw / bh
            if not (self.aspect_min <= aspect <= self.aspect_max):
                continue

            area_ratio = area / img_area
            bbox_ratio = (bw * bh) / img_area
            if area_ratio > self.max_area_ratio or bbox_ratio > self.max_area_ratio:
                continue

            rectangularity = area / (bw * bh)
            cx_norm = (x + bw / 2) / w
            cy_norm = (y + bh / 2) / h
            center_bias = (1.0 - abs(cx_norm - 0.5)) * (1.0 - max(cy_norm - 0.7, 0.0))

            score = area_ratio * rectangularity * center_bias
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)

        if best is None:
            return None

        x, y, bw, bh = best
        roi = gray[y : y + bh, x : x + bw]
        if roi.size > 0:
            lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            if lap_var < self.min_std:
                log.debug(f"  CV detector: rejected low-sharpness region (lap_var={lap_var:.1f})")
                return None

        log.debug(f"  CV detector: bbox={best}, score={best_score:.3f}")
        return best


# ─── 2b. SAM1 detector ────────────────────────────────────────────────────────

class SAMSlideDetector(SlideDetector):
    """
    SAM1-based slide detector using automatic mask generation.

    pip install segment-anything
    Checkpoint: sam_vit_h_4b8939.pth  (or vit_l / vit_b variants)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        min_area_ratio: float = 0.08,
        aspect_min: float = 1.1,
        aspect_max: float = 2.6,
        points_per_side: int = 16,
    ):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.min_area_ratio = min_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        self.mask_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=1000,
        )
        log.info(f"SAM1 loaded: {model_type} on {device}")

    def _score(self, mask: dict, img_h: int, img_w: int) -> float:
        seg: np.ndarray = mask["segmentation"]
        area_ratio = float(seg.sum()) / (img_h * img_w)
        if area_ratio < self.min_area_ratio:
            return 0.0
        x, y, bw, bh = mask["bbox"]
        if bh == 0:
            return 0.0
        aspect = bw / bh
        if not (self.aspect_min <= aspect <= self.aspect_max):
            return 0.0
        rectangularity = float(seg.sum()) / max(bw * bh, 1)
        cx_norm = (x + bw / 2) / img_w
        cy_norm = (y + bh / 2) / img_h
        center_bias = (1.0 - abs(cx_norm - 0.5)) * (1.0 - max(cy_norm - 0.7, 0.0))
        return area_ratio * rectangularity * center_bias

    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        masks = self.mask_gen.generate(frame_rgb)
        if not masks:
            return None
        best = max(masks, key=lambda m: self._score(m, h, w))
        if self._score(best, h, w) < 0.01:
            return None
        x, y, bw, bh = (int(v) for v in best["bbox"])
        return (x, y, bw, bh)


# ─── 2c. SAM2 detector ────────────────────────────────────────────────────────

class SAM2SlideDetector(SlideDetector):
    """
    SAM2-based slide detector (Meta's Segment Anything Model 2, 2024).

    pip install sam2   (or install from https://github.com/facebookresearch/sam2)
    Checkpoints: sam2_hiera_large.pt, sam2_hiera_base_plus.pt, etc.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
        min_area_ratio: float = 0.08,
        aspect_min: float = 1.1,
        aspect_max: float = 2.6,
        points_per_side: int = 16,
    ):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        self.min_area_ratio = min_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

        sam2 = build_sam2(config_path, checkpoint_path, device=device)
        self.mask_gen = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=1000,
        )
        log.info(f"SAM2 loaded: {config_path} on {device}")

    def _score(self, mask: dict, img_h: int, img_w: int) -> float:
        seg: np.ndarray = mask["segmentation"]
        area_ratio = float(seg.sum()) / (img_h * img_w)
        if area_ratio < self.min_area_ratio:
            return 0.0
        x, y, bw, bh = mask["bbox"]
        if bh == 0:
            return 0.0
        aspect = bw / bh
        if not (self.aspect_min <= aspect <= self.aspect_max):
            return 0.0
        rectangularity = float(seg.sum()) / max(bw * bh, 1)
        cx_norm = (x + bw / 2) / img_w
        cy_norm = (y + bh / 2) / img_h
        center_bias = (1.0 - abs(cx_norm - 0.5)) * (1.0 - max(cy_norm - 0.7, 0.0))
        return area_ratio * rectangularity * center_bias

    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        masks = self.mask_gen.generate(frame_rgb)
        if not masks:
            return None
        best = max(masks, key=lambda m: self._score(m, h, w))
        if self._score(best, h, w) < 0.01:
            return None
        x, y, bw, bh = (int(v) for v in best["bbox"])
        return (x, y, bw, bh)


# ─── 2d. Grounded SAM 2 (text-prompt → bbox via GDINO → mask via SAM2) ────────

class GroundedSAM2SlideDetector(SlideDetector):
    """
    Text-prompted slide detector: Grounding DINO → SAM2.

    Install
    -------
    pip install groundingdino-py sam2

    Checkpoints
    -----------
    groundingdino_swint_ogc.pth  +  GroundingDINO_SwinT_OGC.py  (config)
    sam2.1_hiera_large.pt
    """

    def __init__(
        self,
        gdino_config: str,
        gdino_checkpoint: str,
        sam2_checkpoint: str,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        text_prompt: str = "projection screen . presentation slide . powerpoint slide",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        min_area_ratio: float = 0.10,
    ):
        from groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_area_ratio = min_area_ratio
        self.person_prompt = "person . face"
        self.person_box_threshold = 0.4
        self.max_person_coverage = 0.1

        self.gdino = load_model(gdino_config, gdino_checkpoint, device=device)
        sam2 = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2)

        log.info(f"Grounded SAM2 loaded | prompt: {text_prompt!r}")

    @staticmethod
    def _person_coverage(
        slide_box: "torch.Tensor",
        person_boxes: "torch.Tensor",
    ) -> float:
        """Fraction of slide_box area covered by person/face boxes."""
        sx1, sy1, sx2, sy2 = slide_box.tolist()
        slide_area = max((sx2 - sx1) * (sy2 - sy1), 1.0)
        covered = 0.0
        for pb in person_boxes:
            px1, py1, px2, py2 = pb.tolist()
            ix1, iy1 = max(sx1, px1), max(sy1, py1)
            ix2, iy2 = min(sx2, px2), min(sy2, py2)
            if ix2 > ix1 and iy2 > iy1:
                covered += (ix2 - ix1) * (iy2 - iy1)
        return min(covered / slide_area, 1.0)

    def _filter_by_persons(
        self,
        slide_xyxy: "torch.Tensor",
        person_xyxy: "Optional[torch.Tensor]",
    ) -> "torch.Tensor":
        """Drop slide boxes where >max_person_coverage of their area is people.
        Falls back to the least-covered box if all would be dropped.
        """
        import torch

        if person_xyxy is None or len(person_xyxy) == 0:
            return slide_xyxy

        coverages = [
            self._person_coverage(slide_xyxy[i], person_xyxy)
            for i in range(len(slide_xyxy))
        ]
        keep = [i for i, c in enumerate(coverages) if c <= self.max_person_coverage]
        if not keep:
            best = int(np.argmin(coverages))
            log.debug(
                f"  GSAM2: all slide boxes overlap people; "
                f"keeping least-covered ({coverages[best]:.0%})"
            )
            keep = [best]
        elif len(slide_xyxy) - len(keep):
            log.debug(f"  GSAM2: dropped {len(slide_xyxy) - len(keep)} slide box(es) with >50% person overlap")
        return slide_xyxy[torch.tensor(keep)]

    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        import torch
        from groundingdino.util.inference import predict as gdino_predict
        from torchvision.ops import box_convert

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        img_tensor = (
            torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        ).to(self.device)

        boxes_norm, logits, _ = gdino_predict(
            self.gdino,
            img_tensor,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
        )

        if boxes_norm is None or len(boxes_norm) == 0:
            log.debug("  GSAM2: Grounding DINO found no slide boxes")
            return None

        # Detect person/face regions with the same GDINO model
        p_boxes_norm, _, _ = gdino_predict(
            self.gdino, img_tensor,
            self.person_prompt, self.person_box_threshold, self.text_threshold,
        )

        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        slide_xyxy = box_convert(boxes_norm.cpu() * scale, in_fmt="cxcywh", out_fmt="xyxy")
        person_xyxy = (
            box_convert(p_boxes_norm.cpu() * scale, in_fmt="cxcywh", out_fmt="xyxy")
            if p_boxes_norm is not None and len(p_boxes_norm) > 0 else None
        )
        log.debug(
            f"  GSAM2: {len(slide_xyxy)} slide box(es), "
            f"{len(person_xyxy) if person_xyxy is not None else 0} person box(es)"
        )

        # Drop slide boxes that are mostly people, then union survivors
        slide_xyxy = self._filter_by_persons(slide_xyxy, person_xyxy)
        x1 = max(0, int(slide_xyxy[:, 0].min().item()))
        y1 = max(0, int(slide_xyxy[:, 1].min().item()))
        x2 = min(w, int(slide_xyxy[:, 2].max().item()))
        y2 = min(h, int(slide_xyxy[:, 3].max().item()))
        log.debug(f"  GSAM2: union after person filter → ({x1},{y1},{x2},{y2})")

        self.predictor.set_image(frame_rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=False,
        )
        mask = masks[0]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            if ((x2 - x1) * (y2 - y1)) / (h * w) < self.min_area_ratio:
                return None
            return (x1, y1, x2 - x1, y2 - y1)

        row_idxs = np.where(rows)[0]
        col_idxs = np.where(cols)[0]
        r_min, r_max = int(row_idxs[0]), int(row_idxs[-1])
        c_min, c_max = int(col_idxs[0]), int(col_idxs[-1])
        bw, bh = c_max - c_min, r_max - r_min

        if (bw * bh) / (h * w) < self.min_area_ratio:
            log.debug(f"  GSAM2: rejected small mask ({bw}x{bh} = {bw*bh/(h*w):.2%} of frame)")
            return None

        log.debug(f"  GSAM2: mask bbox ({c_min},{r_min},{bw},{bh}), score={scores[0]:.2f}")
        return (c_min, r_min, bw, bh)

    def detect_batch(self, frames_bgr: List[np.ndarray]) -> List[Optional[BBox]]:
        """
        GDINO runs sequentially (SwinT encoder); SAM2 image encoding is batched
        via set_image_batch + predict_batch (the actual ViT-H bottleneck).
        """
        import torch
        from groundingdino.util.inference import predict as gdino_predict
        from torchvision.ops import box_convert

        N = len(frames_bgr)
        results: List[Optional[BBox]] = [None] * N

        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        shapes = [f.shape[:2] for f in frames_bgr]

        valid_indices: List[int] = []
        valid_boxes_xyxy: List[List[int]] = []

        for i, (rgb, (h, w)) in enumerate(zip(frames_rgb, shapes)):
            img_t = (
                torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            ).to(self.device)
            boxes_norm, _, _ = gdino_predict(
                self.gdino, img_t,
                self.text_prompt, self.box_threshold, self.text_threshold,
            )
            if boxes_norm is None or len(boxes_norm) == 0:
                continue
            p_boxes_norm, _, _ = gdino_predict(
                self.gdino, img_t,
                self.person_prompt, self.person_box_threshold, self.text_threshold,
            )
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            slide_xyxy = box_convert(boxes_norm.cpu() * scale, in_fmt="cxcywh", out_fmt="xyxy")
            person_xyxy = (
                box_convert(p_boxes_norm.cpu() * scale, in_fmt="cxcywh", out_fmt="xyxy")
                if p_boxes_norm is not None and len(p_boxes_norm) > 0 else None
            )
            slide_xyxy = self._filter_by_persons(slide_xyxy, person_xyxy)
            x1 = max(0, int(slide_xyxy[:, 0].min().item()))
            y1 = max(0, int(slide_xyxy[:, 1].min().item()))
            x2 = min(w, int(slide_xyxy[:, 2].max().item()))
            y2 = min(h, int(slide_xyxy[:, 3].max().item()))
            valid_indices.append(i)
            valid_boxes_xyxy.append([x1, y1, x2, y2])

        if not valid_indices:
            return results

        valid_frames_rgb = [frames_rgb[i] for i in valid_indices]
        self.predictor.set_image_batch(valid_frames_rgb)

        box_batch = [np.array(b, dtype=np.float32) for b in valid_boxes_xyxy]
        masks_batch, scores_batch, _ = self.predictor.predict_batch(
            point_coords_batch=None,
            point_labels_batch=None,
            box_batch=box_batch,
            multimask_output=False,
        )

        for frame_idx, (h, w), (x1, y1, x2, y2), masks, scores in zip(
            valid_indices, [shapes[i] for i in valid_indices],
            valid_boxes_xyxy, masks_batch, scores_batch,
        ):
            mask = masks[0]
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not rows.any():
                bw, bh = x2 - x1, y2 - y1
                if (bw * bh) / (h * w) >= self.min_area_ratio:
                    results[frame_idx] = (x1, y1, bw, bh)
                continue

            row_idxs = np.where(rows)[0]
            col_idxs = np.where(cols)[0]
            r_min, r_max = int(row_idxs[0]), int(row_idxs[-1])
            c_min, c_max = int(col_idxs[0]), int(col_idxs[-1])
            bw, bh = c_max - c_min, r_max - r_min

            if (bw * bh) / (h * w) < self.min_area_ratio:
                log.debug(f"  GSAM2 batch: rejected small mask ({bw}x{bh})")
                continue

            results[frame_idx] = (c_min, r_min, bw, bh)

        return results


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_detector(
    mode: str,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_h",
    sam2_config: str = "sam2_hiera_l.yaml",
    device: str = "cuda",
    min_area_ratio: float = 0.10,
    gdino_config: Optional[str] = None,
    gdino_checkpoint: Optional[str] = None,
    gdino_text_prompt: str = "projection screen . presentation slide . powerpoint slide",
    gdino_box_threshold: float = 0.30,
) -> SlideDetector:
    """Factory: 'cv' | 'sam' | 'sam2' | 'gsam2'."""
    if mode == "cv":
        return CVSlideDetector(min_area_ratio=min_area_ratio)
    if mode == "sam":
        if not sam_checkpoint:
            raise ValueError("--sam-checkpoint required for --detector sam")
        return SAMSlideDetector(sam_checkpoint, sam_model_type, device,
                                min_area_ratio=min_area_ratio)
    if mode == "sam2":
        if not sam_checkpoint:
            raise ValueError("--sam-checkpoint required for --detector sam2")
        return SAM2SlideDetector(sam_checkpoint, sam2_config, device,
                                 min_area_ratio=min_area_ratio)
    if mode == "gsam2":
        for name, val in [("--gdino-config", gdino_config),
                          ("--gdino-checkpoint", gdino_checkpoint),
                          ("--sam-checkpoint", sam_checkpoint)]:
            if not val:
                raise ValueError(f"{name} required for --detector gsam2")
        return GroundedSAM2SlideDetector(
            gdino_config=gdino_config,
            gdino_checkpoint=gdino_checkpoint,
            sam2_checkpoint=sam_checkpoint,
            sam2_config=sam2_config,
            device=device,
            text_prompt=gdino_text_prompt,
            box_threshold=gdino_box_threshold,
            min_area_ratio=min_area_ratio,
        )
    raise ValueError(f"Unknown detector: {mode!r}. Choose cv | sam | sam2 | gsam2")
