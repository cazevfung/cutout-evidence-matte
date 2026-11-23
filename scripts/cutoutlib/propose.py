"""R4 三路提案的融合（只产出候选 mask，不产出像素）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import imaging
from .cache import Proposals

SPLASH_KEYS = ("水花", "浪花", "水沫", "水晕", "水波", "光斑", "气泡", "特效", "烟雾", "火")
UI_KEYS = ("图标", "按键", "按钮", "UI", "ui", "提示")
FISH_KEYS = ("鱼",)
THIN_KEYS = ("牵引线", "风筝线", "鱼线", "细线", "绳索", "线缆", "钓鱼线")
SUBJECT_KEYS = ("少女", "角色", "人物", "人", "主角", "主体", "卡通", "企鹅", "鲸", "风筝", "玩具")


@dataclass
class ProposalSet:
    p1: np.ndarray                      # P1 显著性 alpha (uint8)
    p2: np.ndarray                      # 前景层 alpha
    p3: np.ndarray | None               # 色键 mask（已对齐回原图坐标）
    p3_error: float | None              # 对齐误差 px
    exclusions: dict = field(default_factory=dict)   # name -> {"alpha": bool, "rgb": HxWx3, "color": bool}
    attachments: dict = field(default_factory=dict)  # 该保留的附属物（鱼等）
    labels: dict = field(default_factory=dict)
    p2_nonkeep: np.ndarray | None = None             # P2 明确归入"不是要保留的东西"的像素（背景/道具/雕塑）
    p2_foreground: np.ndarray | None = None          # P2 全部非 base 图层（"这里是某个物体，不是背景"）
    p2_layers: list = field(default_factory=list)    # 每个图层的 kind，给评分卡溯源用


def classify(label: str) -> str:
    for k in SPLASH_KEYS:
        if k in label:
            return "splash"
    for k in UI_KEYS:
        if k in label:
            return "ui"
    for k in SUBJECT_KEYS:
        if k in label:
            return "subject"
    for k in THIN_KEYS:
        if k in label:
            return "thin"
    for k in FISH_KEYS:
        if k in label:
            return "fish"
    return "other"


def chroma_key(key_rgb: np.ndarray, green=(0, 255, 0), tol: int = 90) -> np.ndarray:
    d = np.abs(key_rgb.astype(np.int16) - np.array(green, np.int16)).sum(axis=2)
    return d > tol


def layer_rgb_on(layer, shape) -> np.ndarray:
    """把图层的 RGB 按 bbox 缩放贴回原图坐标（用于"这块到底长什么样"的判断）。"""
    import cv2

    rgba = imaging.load_rgba(layer.file)
    x0, y0, x1, y1 = layer.bbox
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    small = cv2.resize(rgba[..., :3], (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((shape[0], shape[1], 3), np.uint8)
    hh = min(h, shape[0] - y0)
    ww = min(w, shape[1] - x0)
    if hh > 0 and ww > 0:
        out[y0:y0 + hh, x0:x0 + ww] = small[:hh, :ww]
    return out


def build(case, prop: Proposals, use_p3: bool = True, src_rgb: np.ndarray | None = None,
          color_tol: float = 55.0) -> ProposalSet:
    shape = None
    p1 = prop.bria
    if p1 is None:
        raise ValueError("采集缓存里没有 P1（bria.png）")
    shape = p1.shape

    p2 = np.zeros(shape, np.uint8)
    foreground = np.zeros(shape, np.uint8)
    nonkeep = np.zeros(shape, bool)
    exclusions: dict[str, np.ndarray] = {}
    attachments: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for layer in prop.layers:
        if layer.name == "base":
            continue
        a = layer.alpha_on(shape)
        foreground = np.maximum(foreground, a)
        kind = classify(layer.label or layer.name)
        labels[layer.name] = kind
        if kind in ("subject", "thin"):
            p2 = np.maximum(p2, a)
        elif kind in ("splash", "ui"):
            entry = exclusions.setdefault(kind, {"alpha": np.zeros(shape, bool),
                                                 "rgb": np.zeros(shape + (3,), np.uint8),
                                                 "color": np.zeros(shape, bool)})
            entry["alpha"] |= a > 40
            rgb_layer = layer_rgb_on(layer, shape)
            paint = a > 40
            entry["rgb"][paint] = rgb_layer[paint]
            if src_rgb is not None:
                # 只把"长得像这一层"的像素当排除项：水花是白的、鱼是深蓝的，
                # 同处一个 bbox 也不会被一起删掉
                same = np.linalg.norm(src_rgb.astype(np.float32) - rgb_layer.astype(np.float32),
                                      axis=2) < color_tol
                entry["color"] |= paint & same
                entry["alpha"] = paint
                # 颜色证据优先：没有 RGB 证据的地方不排（宁可留一点水花，也不删掉鱼）
                entry["mask"] = entry["color"] if entry["color"].any() else paint
            else:
                entry["mask"] = paint
        elif kind == "fish":
            attachments[layer.name] = a > 40
        elif kind == "other":
            # 背景/道具/雕塑/房屋这类"明确不要"的层：P2 对这块有意见，融合时用它做减法
            nonkeep |= a > 64

    p3 = None
    p3_err = None
    if use_p3 and prop.key_rgb is not None:
        kf = chroma_key(prop.key_rgb)
        # 绿幕前景是 16:9 的另一画幅，先用 (sx,sy) 网格对回原图
        p3, p3_err = _align_by_scale(kf, p1 > 0)
        if p3_err > 4.0:
            p3 = None
    for name, entry in exclusions.items():
        entry.setdefault("mask", entry["alpha"] if src_rgb is None else entry.get("color", entry["alpha"]))
    return ProposalSet(p1=p1, p2=p2, p3=p3, p3_error=p3_err,
                       exclusions=exclusions, attachments=attachments, labels=labels,
                       p2_nonkeep=nonkeep, p2_foreground=foreground,
                       p2_layers=[{"name": k, "kind": v} for k, v in labels.items()])


def _align_by_scale(key_fg: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """把绿幕前景对回原图坐标：搜 (sx,sy,tx,ty)，使 ref 放大后与 key 前景重合。

    返回 (原图坐标系下的 P3 mask, 对齐误差 px)。误差 = 两侧边界像素到对侧边界的平均距离。
    """
    import cv2

    Hs, Ws = ref.shape
    Hk, Wk = key_fg.shape
    kd = cv2.resize(key_fg.astype(np.uint8), (Wk // 2, Hk // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    refu = ref.astype(np.uint8)
    best = (-1.0, None)
    for step, center, span in ((0.04, None, None), (0.01, None, None)):
        if best[1] is None:
            sx0, sy0 = 1.10, 1.10
            n = int((1.70 - 1.10) / step)
            grid = [(1.10 + i * step, 1.10 + j * step) for i in range(n + 1) for j in range(n + 1)]
        else:
            base_sx, base_sy = best[1][0], best[1][1]
            grid = [(base_sx + i * step, base_sy + j * step) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        for sx, sy in grid:
            w, h = int(Ws * sx / 2), int(Hs * sy / 2)
            if not (8 <= w < kd.shape[1] and 8 <= h < kd.shape[0]):
                continue
            templ = cv2.resize(refu, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            if templ.sum() < 10:
                continue
            res = cv2.matchTemplate(kd, templ, cv2.TM_CCOEFF)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, (sx, sy, loc[0] * 2, loc[1] * 2))
    if best[1] is None:
        return np.zeros(ref.shape, bool), 999.0
    sx, sy, tx, ty = best[1]
    # 原图点 p 对应绿幕点 q = (sx*p.x + tx, sy*p.y + ty)，所以把绿幕缩 1/s 后平移 -t/s 贴回
    sw, sh = max(int(round(Wk / sx)), 1), max(int(round(Hk / sy)), 1)
    small = cv2.resize(key_fg.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    offx, offy = int(round(-tx / sx)), int(round(-ty / sy))
    out = np.zeros(ref.shape, bool)
    x0, y0 = max(offx, 0), max(offy, 0)
    x1, y1 = min(offx + sw, Ws), min(offy + sh, Hs)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = small[y0 - offy:y1 - offy, x0 - offx:x1 - offx]

    def _edge(m):
        m8 = m.astype(np.uint8)
        return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0

    e_out, e_ref = _edge(out), _edge(ref)
    if not e_out.any() or not e_ref.any():
        return out, 999.0
    d_to_out = cv2.distanceTransform((~e_out).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_ref = cv2.distanceTransform((~e_ref).astype(np.uint8), cv2.DIST_L2, 5)
    # 用中位数而不是均值：参照里多出来的物体（鱼）会把均值拉到 50px 以上，中位数才是"对齐误差"
    err = float(np.median(np.r_[d_to_ref[e_out], d_to_out[e_ref]]))
    return out, err


def exclusion_masks(ps: ProposalSet) -> dict:
    return {k: v.get("mask", v["alpha"]) for k, v in ps.exclusions.items()}


def fuse(ps: ProposalSet, params: dict) -> np.ndarray:
    """融合成一张 0–255 的 mask（不是最终 alpha）。

    规则（T0 结论）：P1 为骨架；P2 只补与 P1 相邻的连通块；P3 只在 P1∪P2 整体缺失时补位；
    排除区（水花/UI）在最终 mask 里强制清零。
    """
    p1 = (ps.p1 >= int(params.get("p1_thr", 128)))
    core = p1.copy()

    p2 = ps.p2 >= int(params.get("p2_thr", 128))
    if params.get("use_p2", True):
        near = imaging.dilate(core, int(params.get("p2_join_px", 3)))
        add = p2 & near & ~core
        n, lab, stats, _ = imaging.label(add)
        min_area = int(params.get("p2_min_area", 40))
        keep = np.zeros_like(add)
        for i in range(1, n):
            if stats[i, cv2_CC_AREA] >= min_area:
                keep |= lab == i
        core |= keep

    # P2 主体层优先：P1（显著性）分不清"显眼的背景物件"和"主体"——显卡盒上的金属底座、
    # 印在盒面上的亮色文字，都会被它算成主体。图层拆分把画面分成 base（背景）+ 若干物体层；
    # 落在 base 里的 P1 像素就是背景，一律剔除；落在任何一个物体层里的（哪怕那层不是本次
    # 要保留的东西）保留，因为"不要它"是排除区该管的事，不由这条规则决定。黑色过膝袜压在
    # 黑色盒面上这类同色区，只有 P2 能给出正确轮廓，所以 P2 认下的一律保留。
    if params.get("p2_authority", False):
        auth = ps.p2_foreground if ps.p2_foreground is not None else p2
        if auth.any():
            ref = auth >= int(params.get("p2_authority_thr", 128))
            # P2 认下、但 P1 完全没碰到的远处块不强行加：那只说明 P1 没检到它，不代表
            # 本次要保留（它的去留由 keep/exclude 与细结构通道决定）。
            near_p1 = imaging.dilate(p1 >= int(params.get("p1_thr", 128)),
                                     int(params.get("p2_authority_add_px", 16)))
            core = (core & imaging.dilate(ref, int(params.get("p2_authority_px", 4)))) \
                | (p2 & near_p1)

    p3 = ps.p3
    if p3 is not None and params.get("use_p3", False):
        near = imaging.dilate(core, int(params.get("p3_join_px", 6)))
        add = p3 & ~imaging.dilate(core, 1)
        n, lab, stats, _ = imaging.label(add)
        for i in range(1, n):
            comp = lab == i
            if stats[i, cv2_CC_AREA] >= int(params.get("p3_min_area", 200)) and (comp & near).any():
                core |= comp

    # P2 否决权：某像素被 P2 明确归入"非 keep 层"、且不落在任何 keep 层里 → 从融合结果剔除。
    # P1（显著性）只在 P2 没意见的地方保留主导权。这条是通用的，不看题材。
    if params.get("p2_veto", True) and ps.p2_nonkeep is not None and ps.p2_nonkeep.any():
        keep_claim = p2 | (ps.p2 >= int(params.get("p2_thr", 128)))
        for att in ps.attachments.values():
            keep_claim |= att
        veto = ps.p2_nonkeep & ~imaging.dilate(keep_claim, int(params.get("veto_protect_px", 1)))
        core = core & ~veto

    if params.get("fill_holes", True):
        core = imaging.fill_holes(core)

    # 附属物（鱼）：与 core 相邻才并，并了以后不再被排除区整块清掉
    if params.get("keep_attachments", True):
        for name, att in ps.attachments.items():
            x0, y0, x1, y1 = imaging.bbox_of(att)
            if (x1 - x0) * (y1 - y0) / (core.shape[0] * core.shape[1]) > params.get("attach_max_frac", 0.05):
                continue
            if (imaging.dilate(core, int(params.get("attach_join_px", 6))) & att).any():
                core |= att

    # 排除区（保护 subject / 附属物 / 细结构：排除项只吃"没有语义证据"的那些像素）
    ex_all = np.zeros(core.shape, bool)
    for name, ex in exclusion_masks(ps).items():
        ex_all |= ex
    if params.get("use_exclusions", True) and ex_all.any():
        margin = int(params.get("exclusion_margin", 2))
        protected = np.zeros(core.shape, bool)
        if params.get("protect_semantics", True):
            protected |= ps.p2 >= int(params.get("p2_thr", 128))
            for att in ps.attachments.values():
                protected |= att
        core &= ~(imaging.dilate(ex_all, margin) & ~protected)

    # 最小面积与连通性
    n, lab, stats, _ = imaging.label(core)
    if n > 1:
        areas = stats[1:, cv2_CC_AREA]
        biggest = areas.max()
        min_frac = float(params.get("min_area_frac", 0.0005))
        thr = max(min_frac * core.size, 1)
        keep = np.zeros_like(core)
        for i in range(1, n):
            a = stats[i, cv2_CC_AREA]
            if a >= thr and a >= biggest * float(params.get("min_rel_area", 0.02)):
                keep |= lab == i
        core = keep
    return (core.astype(np.uint8) * 255)


cv2_CC_AREA = 4  # cv2.CC_STAT_AREA，避免在模块顶层 import cv2
