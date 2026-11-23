"""采集层缓存契约。

采集可以由任意工具完成（也可以完全手写），决策层只读磁盘上的产物。
产物按下面的形状落在 `<cache-root>/<case>/<recipe_tag>/`；本模块只负责**读**。

    <cache-root>/<case>/<tag>/
        meta.json      # {case_id, bria, key, layers:[{z,name,label,bbox,file}]}
        bria.png       # P1：显著性去背景 RGBA
        key.png        # P3：绿幕编辑图（RGB，纯 #00FF00 底）
        layers/z1_x.png ...   # P2：图层拆分（RGBA，与 bbox 同尺寸，按 bbox 贴回原图）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import imaging


class CacheMiss(Exception):
    """采集层缺件。"""


@dataclass
class Layer:
    z: int
    name: str
    label: str
    bbox: tuple[int, int, int, int]
    file: Path

    def alpha_on(self, shape: tuple[int, int]) -> np.ndarray:
        rgba = imaging.load_rgba(self.file)
        x0, y0, x1, y1 = self.bbox
        w, h = max(x1 - x0, 1), max(y1 - y0, 1)
        a = imaging.resize_alpha(rgba[..., 3], (w, h))
        out = np.zeros(shape, np.uint8)
        hh = min(h, shape[0] - y0)
        ww = min(w, shape[1] - x0)
        if hh <= 0 or ww <= 0:
            return out
        out[y0:y0 + hh, x0:x0 + ww] = a[:hh, :ww]
        return out


@dataclass
class Proposals:
    case_id: str
    tag: str
    root: Path
    bria: np.ndarray | None = None            # HxW uint8 alpha
    key_rgb: np.ndarray | None = None         # HxWx3 RGB
    layers: list[Layer] = field(default_factory=list)
    capture_calls: dict = field(default_factory=dict)
    acquired_at: str = ""

    def has(self, what: str) -> bool:
        return {
            "bria": self.bria is not None,
            "key": self.key_rgb is not None,
            "layers": bool(self.layers),
        }[what]


def cache_root(root: str | Path | None = None) -> Path:
    """采集缓存根目录。默认当前目录下的 `cache`，可用参数或 CUTOUT_CACHE 覆盖。"""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("CUTOUT_CACHE") or "cache")


def cache_dir(case_id: str, tag: str, root: str | Path | None = None) -> Path:
    return cache_root(root) / case_id / tag


def list_tags(case_id: str, root: str | Path | None = None) -> list[str]:
    base = cache_root(root) / case_id
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "meta.json").exists())


def load(case_id: str, tag: str | None = None, root: str | Path | None = None) -> Proposals:
    base = cache_root(root) / case_id
    if not base.exists():
        raise CacheMiss(
            f"没有采集缓存目录：{base.resolve()}。这里应放 bria.png（P1）、key.png（P3）、"
            f"layers/*.png（P2）与 meta.json，见 SKILL.md 的采集层契约")
    if tag is None:
        tags = list_tags(case_id, root)
        if not tags:
            raise CacheMiss(f"缓存目录空：{base.resolve()}（需要 <tag>/meta.json）")
        tag = "auto" if "auto" in tags else tags[0]
    d = base / tag
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise CacheMiss(f"缺 meta.json（三路产物的索引）：{meta_path.resolve()}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # P1 / P2 / P3 逐一校验，缺哪一路就报哪一路，不吐裸栈。
    bria = meta.get("bria")
    if not bria:
        raise CacheMiss("meta.json 没写 bria（P1 显著性提案 bria.png）")
    bria_path = d / bria
    if not bria_path.exists():
        raise CacheMiss(f"缺 P1（显著性提案）：{bria_path.resolve()}")

    key = meta.get("key")
    if not key:
        raise CacheMiss("meta.json 没写 key（P3 绿幕提案 key.png）")
    key_path = d / key
    if not key_path.exists():
        raise CacheMiss(f"缺 P3（绿幕提案）：{key_path.resolve()}")

    items = meta.get("layers") or []
    if not items:
        raise CacheMiss("缺 P2（图层拆分）：meta.json 的 layers 为空，至少要有一层")
    layers: list[Layer] = []
    for item in items:
        f = d / str(item.get("file", ""))
        if not f.exists():
            raise CacheMiss(f"缺 P2 图层文件：{f.resolve()}")
        try:
            bbox = tuple(int(v) for v in item["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheMiss(f"P2 图层 bbox 必须是原图坐标 [x0,y0,x1,y1]：{item.get('name')}") from exc
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise CacheMiss(f"P2 图层 bbox 无效（需要原图坐标 x0,y0,x1,y1 且非空）：{item.get('name')}")
        layers.append(Layer(z=int(item.get("z", 0)), name=str(item.get("name", f.stem)),
                            label=str(item.get("label", "")), bbox=bbox, file=f))

    prop = Proposals(case_id=case_id, tag=tag, root=d,
                     capture_calls=meta.get("capture_calls", {}),
                     acquired_at=meta.get("acquired_at", ""))
    prop.bria = imaging.load_rgba(bria_path)[..., 3]
    prop.key_rgb = imaging.load_rgba(key_path)[..., :3]
    prop.layers = layers
    return prop


def write_meta(case_id: str, tag: str, meta: dict, root: str | Path | None = None) -> Path:
    d = cache_dir(case_id, tag, root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return p
