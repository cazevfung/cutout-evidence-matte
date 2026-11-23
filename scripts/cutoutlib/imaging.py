"""图像基础运算：全是 numpy / opencv，不碰任何模型。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"读不出图片：{Path(path).resolve()}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_rgba(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"读不出图片：{Path(path).resolve()}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def save_rgba(path: str | Path, rgba: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise IOError(f"写不出 PNG：{p}")
    buf.tofile(str(p))


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"写不出 PNG：{p}")
    buf.tofile(str(p))


def save_gray(path: str | Path, gray: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", gray.astype(np.uint8))
    if not ok:
        raise IOError(f"写不出 PNG：{p}")
    buf.tofile(str(p))


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def composite(rgb: np.ndarray, alpha: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)[..., None]
    bg_arr = np.array(bg, np.float32).reshape(1, 1, 3)
    return np.clip(rgb.astype(np.float32) * a + bg_arr * (1.0 - a), 0, 255).astype(np.uint8)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), k) > 0


def label(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (n, labels, stats, centroids)；stats 每行 = x,y,w,h,area。"""
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def components(mask: np.ndarray) -> list[np.ndarray]:
    n, lab, stats, _ = label(mask)
    return [(lab == i) for i in range(1, n)]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return (m | holes) > 0


def skeleton(mask: np.ndarray) -> np.ndarray:
    """形态学骨架（不依赖 opencv-contrib）。"""
    img = mask.astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(4000):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    """每个像素到 mask 边界的距离（mask 外为正，内为正）。"""
    m = mask.astype(np.uint8)
    return np.maximum(cv2.distanceTransform(m, cv2.DIST_L2, 5),
                      cv2.distanceTransform(1 - m, cv2.DIST_L2, 5))


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if (alpha.shape[1], alpha.shape[0]) == (w, h):
        return alpha
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)


def sample_polyline(poly: np.ndarray, step: float = 1.0) -> np.ndarray:
    """沿折线按 step 采样点。"""
    pts = [np.asarray(poly[0], np.float64)]
    for a, b in zip(poly[:-1], poly[1:]):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for i in range(1, n + 1):
            pts.append(a + (b - a) * (i / n))
    return np.asarray(pts)
