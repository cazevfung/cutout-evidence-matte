#!/usr/bin/env python3
"""Regenerate the offline demo assets (source image + three proposal products).

The demo exists to prove that the decision layer is independent of any capture
service: it reads a source image and three ordinary files (P1 / P2 / P3) that a
human can author by hand.  This script draws those files deterministically with
numpy/OpenCV only -- no network, no model, no recorded cloud response.

Run from the package root:

    python scripts/make_demo.py

It overwrites demo/source.png and demo/cache/demo-scene/auto/*.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
CACHE = DEMO / "cache" / "demo-scene" / "auto"
W, H = 360, 270

ORANGE = (225, 135, 70)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def _subject_mask() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (112, 88), 16, 255, -1)                       # head
    cv2.ellipse(m, (118, 142), (22, 34), 0, 0, 360, 255, -1)   # torso
    cv2.line(m, (100, 120), (78, 150), 255, 9)                 # left arm
    cv2.line(m, (134, 120), (156, 150), 255, 9)                # right arm
    cv2.line(m, (108, 172), (100, 198), 255, 10)               # left leg
    cv2.line(m, (128, 172), (138, 198), 255, 10)               # right leg
    cv2.ellipse(m, (99, 201), (9, 5), 0, 0, 360, 255, -1)      # left shoe
    cv2.ellipse(m, (139, 201), (9, 5), 0, 0, 360, 255, -1)     # right shoe
    return m


def _line_mask() -> np.ndarray:
    """1px 白色细线：长宽比远大于 12、局部宽度 1px，分割模型拿不到。"""
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (130, 120), (300, 72), 255, 1, lineType=cv2.LINE_8)
    return m


def _splash_mask() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (300, 200), 16, 255, -1)
    cv2.circle(m, (318, 216), 9, 255, -1)
    cv2.circle(m, (286, 218), 7, 255, -1)
    return m


def _paint(rgb: np.ndarray, mask: np.ndarray, color) -> None:
    rgb[mask > 0] = color


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _crop_rgba(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = _bbox(mask)
    out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    out[..., :3] = rgb[y0:y1, x0:x1]
    out[..., 3] = mask[y0:y1, x0:x1]
    return out


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise SystemExit(f"cannot encode {path}")
    buf.tofile(str(path))


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f"cannot encode {path}")
    buf.tofile(str(path))


def main() -> int:
    if CACHE.exists():
        shutil.rmtree(CACHE)
    DEMO.mkdir(parents=True, exist_ok=True)

    subject, line, splash = _subject_mask(), _line_mask(), _splash_mask()
    keep = ((subject > 0) | (line > 0))

    # ---- 原图：蓝灰背景 + 橙色小人 + 1px 白线 + 白色水花 ----
    rgb = np.zeros((H, W, 3), np.uint8)
    ramp = np.linspace(150.0, 176.0, W, dtype=np.float32)
    rgb[..., 0] = np.clip(ramp, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(ramp * 1.16, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(ramp * 1.30, 0, 255).astype(np.uint8)
    _paint(rgb, subject, ORANGE)
    _paint(rgb, line, WHITE)
    _paint(rgb, splash, WHITE)
    _save_rgb(DEMO / "source.png", rgb)

    # ---- P1：显著性提案（只给最显眼的主体）----
    p1 = np.zeros((H, W, 4), np.uint8)
    p1[..., :3] = rgb
    p1[..., 3] = (subject > 0).astype(np.uint8) * 255
    _save_rgba(CACHE / "bria.png", p1)

    # ---- P2：图层拆分（角色 / 水花 / 背景）----
    layers = []
    layer_specs = [
        (0, "background", "背景", ~keep, rgb),
        (1, "character", "角色：橙色小人（含手中细线）", keep, rgb),
        (2, "splash", "水花", splash > 0, rgb),
    ]
    CACHE.mkdir(parents=True, exist_ok=True)
    for z, name, label, mask, src in layer_specs:
        m = (mask.astype(np.uint8) * 255)
        fname = f"layers/z{z}_{name}.png"
        _save_rgba(CACHE / fname, _crop_rgba(m, src))
        layers.append({"z": z, "name": name, "label": label,
                       "bbox": _bbox(m), "file": fname})

    # ---- P3：绿幕编辑图（对齐回原图后只作补位，主配方默认不启用）----
    key = np.zeros((H, W, 3), np.uint8)
    key[...] = GREEN
    _paint(key, subject, ORANGE)
    _paint(key, line, WHITE)
    _save_rgb(CACHE / "key.png", key)

    meta = {
        "case_id": "demo-scene",
        "recipe": "auto",
        "acquired_at": "2026-11-23T00:00:00+00:00",
        "canvas_size": [W, H],
        "source_size": [W, H],
        "note": "手写产物：三个提案文件由 scripts/make_demo.py 用 numpy/OpenCV 直接画出，"
                "不经过任何采集服务，也不是回放件。",
        "bria": "bria.png",
        "key": "key.png",
        "layers": layers,
    }
    (CACHE / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    (DEMO / "case.json").write_text(json.dumps({
        "id": "demo-scene",
        "source": "source.png",
        "keep": ["角色", "细线"],
        "exclude": ["水花"],
        "thin": ["细线"],
        "expect_shadow": False,
        "notes": {"why": "离线自检样本：1px 白线（长宽比 >12、局部宽度 <3px）必须靠几何重建，"
                          "水花必须被排除，像素必须全部来自原图"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"demo written: {DEMO}")
    print(f"  source.png {W}x{H}")
    print(f"  cache      {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
