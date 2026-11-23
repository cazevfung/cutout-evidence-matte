"""case.json 的载入与校验（R1 输入契约）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class CaseError(Exception):
    """case 目录不合契约。"""


@dataclass
class Case:
    id: str
    root: Path
    source: Path
    keep: list[str]
    exclude: list[str] = field(default_factory=list)
    thin: list[str] = field(default_factory=list)
    reference: Path | None = None
    negative_sample: Path | None = None
    expect_shadow: bool | None = None
    notes: dict = field(default_factory=dict)

    @property
    def rel_root(self) -> str:
        return self.root.name

    def out(self, suffix: str) -> Path:
        return Path("out") / f"{self.id}{suffix}"


def load_case(path: str | Path) -> Case:
    root = Path(path)
    if not root.exists():
        raise CaseError(f"case 目录不存在：{root.resolve()}")
    cfg_path = root / "case.json"
    if not cfg_path.exists():
        raise CaseError(f"缺 case.json：{cfg_path.resolve()}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaseError(f"case.json 不是合法 JSON：{cfg_path.resolve()} ({exc})") from exc

    for key in ("id", "source", "keep"):
        if key not in cfg:
            raise CaseError(f"case.json 缺字段 {key}：{cfg_path.resolve()}")
    keep = cfg["keep"]
    if not isinstance(keep, list) or not keep:
        raise CaseError(f"keep 必须是非空列表：{cfg_path.resolve()}")
    source = root / cfg["source"]
    if not source.exists():
        raise CaseError(f"原图不存在：{source.resolve()}")

    ref = cfg.get("reference")
    ref_path = root / ref if ref else None
    if ref_path is not None and not ref_path.exists():
        ref_path = None
    neg = cfg.get("negative_sample")
    neg_path = root / neg if neg else None
    if neg_path is not None and not neg_path.exists():
        neg_path = None

    optional = cfg.get("optional", {}) if isinstance(cfg.get("optional"), dict) else {}
    expect_shadow = cfg.get("expect_shadow", optional.get("expect_shadow"))

    return Case(
        id=str(cfg["id"]),
        root=root,
        source=source,
        keep=[str(x) for x in keep],
        exclude=[str(x) for x in cfg.get("exclude", [])],
        thin=[str(x) for x in cfg.get("thin", [])],
        reference=ref_path,
        negative_sample=neg_path,
        expect_shadow=expect_shadow,
        notes=cfg.get("notes", {}) or {},
    )
