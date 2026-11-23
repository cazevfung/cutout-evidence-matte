"""R5 边界精修：trimap + 纯本地颜色线 matting。

alpha ≈ (I-B)·(F-B) / |F-B|²

F / B 用局部盒滤波从"已知前景 / 已知背景"里估，不联网、不依赖任何 matting 模型。
|F-B| 太小的像素（前景背景同色，颜色线退化）标 low_confidence，不许静默当成功。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imaging


def trimap(mask: np.ndarray, erode_px: int = 2, dilate_px: int = 3) -> np.ndarray:
    """0=确定背景, 128=未知带, 255=确定前景。"""
    fg = imaging.erode(mask > 0, int(erode_px))
    near = imaging.dilate(mask > 0, int(dilate_px))
    tri = np.full(mask.shape, 128, np.uint8)
    tri[~near] = 0
    tri[fg] = 255
    return tri


def _box(arr: np.ndarray, r: int) -> np.ndarray:
    k = (2 * r + 1, 2 * r + 1)
    return cv2.blur(arr.astype(np.float32), k, borderType=cv2.BORDER_REFLECT)


def estimate(rgb: np.ndarray, tri: np.ndarray, params: dict | None = None) -> dict:
    params = params or {}
    r = int(params.get("fb_radius", 9))
    i = rgb.astype(np.float32)
    is_fg = (tri == 255).astype(np.float32)
    is_bg = (tri == 0).astype(np.float32)

    wf = _box(is_fg, r) + 1e-6
    wb = _box(is_bg, r) + 1e-6
    F = _box(i * is_fg[..., None], r) / wf[..., None]
    B = _box(i * is_bg[..., None], r) / wb[..., None]

    fb = F - B
    denom = (fb * fb).sum(axis=2)
    num = ((i - B) * fb).sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denom > 1e-6, num / np.maximum(denom, 1e-6), 0.5)
    alpha = np.clip(alpha, 0.0, 1.0)

    low_conf = denom < float(params.get("min_fb_energy", 900.0))   # |F-B| < 30

    # 颜色线退化时退回几何 alpha：以距离场的平滑过渡为准
    if low_conf.any():
        d = imaging.boundary_distance(tri > 0).astype(np.float32)
        geo = np.clip(0.5 + (d - 1.0) / max(2.0, float(params.get("geo_soft", 3.0))), 0.0, 1.0)
        alpha = np.where(low_conf, geo, alpha)

    alpha[tri == 255] = np.maximum(alpha[tri == 255], 1.0)
    alpha[tri == 0] = 0.0
    # 轻微平滑，避免单像素跳变（3x3，不动已知区）
    soft = cv2.GaussianBlur(alpha, (3, 3), 0)
    unknown = tri == 128
    alpha = np.where(unknown, soft, alpha)
    return {"alpha": alpha.astype(np.float32), "low_conf": low_conf, "F": F, "B": B}


def refine(mask: np.ndarray, rgb: np.ndarray, params: dict | None = None) -> dict:
    params = params or {}
    if not params.get("matte_on", True):
        a = (mask > 0).astype(np.float32)
        return {"alpha": a, "low_conf": np.zeros_like(a, bool), "trimap": (mask > 0).astype(np.uint8) * 255}
    tri = trimap(mask, int(params.get("trimap_erode", 2)), int(params.get("trimap_dilate", 3)))
    res = estimate(rgb, tri, params)
    res["trimap"] = tri
    return res
