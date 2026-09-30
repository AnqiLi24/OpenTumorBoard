from __future__ import annotations

import cv2
import numpy as np

from .detectors import BBox


def crop_slide(
    frame_bgr: np.ndarray,
    bbox: BBox,
    padding: float = 0.005,
) -> np.ndarray:
    """Crop with a small outward padding, then try perspective-rectify.
    Falls back to simple crop if rectification fails.
    """
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    px, py = int(bw * padding), int(bh * padding)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
    cropped = frame_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame_bgr
    return _try_rectify(cropped)


def _try_rectify(img: np.ndarray) -> np.ndarray:
    """Attempt perspective correction by finding the largest quadrilateral in
    the edge map. Returns rectified image, or the original if no good quad found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    best_quad = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.70:  # quad must fill ≥70% of crop — avoids locking onto embedded images (CT, charts)
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return img

    src = best_quad.reshape(4, 2).astype(np.float32)
    src = _order_points(src)
    tw = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
    th = max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (int(tw), int(th)))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
