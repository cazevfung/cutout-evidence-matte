"""一条命令的流水线：原图 + case.json → RGBA / mask / 评分卡 / 配方 / 三底对照图。

顺序固定：缓存（云端提案）→ 融合 → 排除区 → 几何先验 → 颜色线 matting → 细结构 → 评分。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import bgmodel, cache, filter as filt, geo, imaging, matte, propose, recipes, score, thin

KEEP_SUBJECT_KEYS = ("角色", "人物", "少女", "主角", "主体", "人")
KEEP_ROD_KEYS = ("竿",)
KEEP_LINE_KEYS = ("线", "绳")


def _proof(src_rgb: np.ndarray, alpha: np.ndarray, path: Path) -> None:
    panels = []
    for bg in ((255, 0, 255), (250, 250, 250), None):
        if bg is None:
            h, w = alpha.shape
            yy, xx = np.mgrid[0:h, 0:w]
            chk = (((yy // 24) + (xx // 24)) % 2).astype(np.uint8)
            board = np.where(chk[..., None] == 0, 210, 255).astype(np.uint8)
            a = np.clip(alpha, 0, 1)[..., None]
            comp = (src_rgb * a + board * (1 - a)).astype(np.uint8)
        else:
            comp = imaging.composite(src_rgb, alpha, bg)
        panels.append(comp)
    strip = np.concatenate(panels, axis=1)
    small = imaging.resize_alpha(strip, (strip.shape[1] // 3, strip.shape[0] // 3)) if strip.shape[1] > 3000 else strip
    imaging.save_rgb(path, small)


def build_regions(case, ps, rod_line: dict, shape) -> tuple[dict, list[str], list[dict]]:
    """把 keep 清单映射成"应当覆盖的区域"，认不出的项记进 unresolved（AC11 的来源）。"""
    regions: dict[str, np.ndarray] = {}
    unresolved: list[str] = []
    blocks: list[dict] = []

    subject = (ps.p2 > 128)
    rod = rod_line.get("rod")
    line = rod_line.get("line")

    rod_band = None
    if rod is not None:
        rod_band = thin.band(shape, rod.p0, rod.p1, width=max(rod.width * 2.0, 3.0), soft=0.8) > 0.2
        blocks.append({"block_id": "rod", "kind": "thin_band", "source": "geometry:rod",
                       "bbox": list(imaging.bbox_of(rod_band)), "pixels": int(rod_band.sum()),
                       "reason": "竿身在原图是 2–4px 细条，语义分割只给断续点，按暗细结构几何重建"})
    line_band = None
    if line is not None:
        # 1px 细线的"位置"就是它的中心线；用带子量覆盖率会在边缘上系统性偏低
        line_band = thin.band(shape, line.p0, line.p1, width=1.0, soft=0.0) > 0.2
        blocks.append({"block_id": "line", "kind": "thin_band", "source": "geometry:line",
                       "bbox": list(imaging.bbox_of(line_band)), "pixels": int(line_band.sum()),
                       "reason": "鱼线是贴亮水面的 1px 白线，原图真实存在但不可语义分割，按亮细结构几何重建"})

    for item in case.keep:
        hit = False
        if any(k in item for k in KEEP_ROD_KEYS) and rod_band is not None:
            regions[item] = rod_band
            hit = True
        if any(k in item for k in KEEP_LINE_KEYS) and line_band is not None:
            regions[item] = line_band
            hit = True
        if not hit and any(k in item for k in KEEP_SUBJECT_KEYS) and subject.any():
            regions[item] = subject
            hit = True
        if not hit and subject.any() and not regions:
            # 兜底：keep 只写了"人+竿"这类复合短语
            regions[item] = subject
            hit = True
        if not hit:
            unresolved.append(item)
    return regions, unresolved, blocks


def exclude_regions(ps) -> dict:
    out = {}
    for name, mask in propose.exclusion_masks(ps).items():
        if mask.any():
            out[name] = mask
    return out


def run_case(case, recipe: dict, cache_tag: str | None = None, write: bool = True,
             out_dir: str | Path = "out", thresholds: dict | None = None,
             reference: np.ndarray | None = None, expect_fail: bool = False,
             cache_root: str | Path | None = None) -> dict:
    src = imaging.load_rgb(case.source)
    shape = src.shape[:2]
    # 包级适配：采集产物不在仓库的 runs/cache 下，由调用方给出缓存根目录
    prop = cache.load(case.id, cache_tag, root=cache_root)
    ps = propose.build(case, prop, use_p3=bool(recipe.get("use_p3", False)), src_rgb=src)

    # 填洞交给 bgmodel：先看洞里像不像背景（"围出来的真实缝隙"不填）。这里把 fuse 的无条件填洞关掉。
    fused = propose.fuse(ps, dict(recipe, use_exclusions=False, fill_holes=False)) > 127
    protected = (ps.p2 >= int(recipe.get("p2_thr", 128)))
    for att in ps.attachments.values():
        protected |= att
    core, dropped = filt.apply(fused, propose.exclusion_masks(ps) if recipe.get("use_exclusions", True) else {},
                               recipe, protected=protected)

    if int(recipe.get("close_r", 0)) > 0:
        import cv2 as _cv2
        r = int(recipe["close_r"])
        k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        core = _cv2.morphologyEx(core.astype("uint8"), _cv2.MORPH_CLOSE, k) > 0

    # 背景色模型：非 keep 层 + 四边框带 → Lab 聚类中心
    lab = bgmodel.to_lab(src)
    bg_sample = bgmodel.sample(ps, shape)
    bg_centers = bgmodel.palette(lab, bg_sample["mask"], k=int(recipe.get("bg_k", 12)))
    bg_dist = bgmodel.distance(lab, bg_centers)

    hole_log = []
    gap_kept = np.zeros(shape, bool)
    if recipe.get("hole_color_check", True):
        core, hole_log = bgmodel.fill_holes_by_color(
            core, lab, bg_dist, tol=float(recipe.get("hole_tol", bgmodel.HOLE_TOL)),
            min_area=int(recipe.get("hole_min_area", bgmodel.HOLE_MIN_AREA)),
            max_fill_px=max(1500, int(float(recipe.get("hole_max_fill_frac", 0.0)) * core.size)))
        # 判定为"看穿的缝隙"且完全在画面内的洞：交给 matte 当已知背景。
        # 否则 matte 会在洞的最内一圈拉出半透 alpha，贴深底看就是一条 1px 亮边。
        gaps = imaging.fill_holes(core) & ~core
        if gaps.any():
            ng, lab_g, st_g, _ = imaging.label(gaps)
            for i in range(1, ng):
                gx, gy, gw, gh, _a = st_g[i]
                if gx > 0 and gy > 0 and gx + gw < shape[1] and gy + gh < shape[0]:
                    gap_kept |= lab_g == i
    elif recipe.get("fill_holes", True):
        core = imaging.fill_holes(core)

    shafts = geo.find_shafts(src, fused, recipe, nonkeep=ps.p2_nonkeep)
    rod_line = geo.rod_and_line(shafts, fused)
    path = geo.path_points(rod_line, shape)

    # 细结构带也算"有证据"，参与 matte 的已知前景
    mres = matte.refine(core, src, recipe, known_bg=gap_kept)
    alpha = mres["alpha"]

    thin_info = []
    estimated = []
    already_est = np.zeros(shape, bool)
    if recipe.get("thin_on", True):
        for shaft in shafts[:2]:
            if shaft.kind not in ("rod", "line"):
                continue
            tb = thin.build(src, shaft, recipe)
            a = np.maximum(alpha, tb["alpha"])
            alpha = a
            thin_info.append({"kind": shaft.kind, "support": tb["support"],
                              "p0": tb["p0"], "p1": tb["p1"], "width": tb["width"]})
            est = (tb["alpha"] > 0.05) & ~already_est
            already_est |= est
            estimated.append({"block_id": f"thin:{shaft.kind}", "bbox": list(imaging.bbox_of(est)),
                              "pixels": int(est.sum()), "reason": "细结构几何重建（R6）",
                              "support": tb["support"], "kind": shaft.kind})

    # 背景色否决：alpha>0 但颜色就是背景、又不在轮廓 2px 软边带里的成片像素强制置 0
    veto_log = []
    if recipe.get("bg_veto", True):
        alpha, veto_log = bgmodel.veto_background(
            alpha, bg_dist, band_px=int(recipe.get("veto_band_px", 2)),
            tol=float(recipe.get("bg_veto_tol", bgmodel.VM_TOL)),
            min_area=int(recipe.get("veto_min_area", bgmodel.VETO_MIN_AREA)),
            p90_tol=float(recipe.get("veto_p90_tol", bgmodel.VETO_P90_TOL)),
            protect=(ps.p2 >= int(recipe.get("p2_thr", 128))))

    regions, unresolved, blocks = build_regions(case, ps, rod_line, shape)
    keep_region_blocks = [{"item": name, "bbox": list(imaging.bbox_of(region)),
                           "pixels": int(region.sum())} for name, region in regions.items()]

    thin_cov = None
    if path.size:
        thin_cov = score.continuity_metric(alpha, path)["coverage"]

    keep_mask = (ps.p2 > 128)
    for att in ps.attachments.values():
        keep_mask |= att > 128
    nonkeep_layers = []
    if ps.p2_nonkeep is not None and ps.p2_nonkeep.any():
        nonkeep_layers.append(ps.p2_nonkeep)

    ctx = {
        "keep_pixels": int(keep_mask.sum()),
        "keep_mask": keep_mask,
        "p1_pixels": int((ps.p1 > 128).sum()),
        "nonkeep_layers": nonkeep_layers,
        "keep_regions": regions,
        "exclude_regions": exclude_regions(ps),
        "path": path,
        "reference": reference if reference is not None else (
            imaging.load_rgba(case.reference) if case.reference else None),
        "homography": None,
        "estimated": estimated,
        "low_conf_pixels": int(mres["low_conf"].sum()),
        "dropped": dropped,
        "recipe": {k: v for k, v in recipe.items()},
        "cache_tag": prop.tag,
        "capture_calls": prop.capture_calls,
        "thin_coverage": thin_cov,
        "thin_structures": thin_info,
        "hole_decisions": hole_log,
        "bg_veto": veto_log,
        "bg_model": {"k": int(recipe.get("bg_k", 12)),
                     "samples_from_layers": bg_sample["from_layers"],
                     "samples_from_border": bg_sample["from_border"],
                     "samples_from_no_object": bg_sample.get("from_base", 0),
                     "hole_tol": float(recipe.get("hole_tol", bgmodel.HOLE_TOL)),
                     "veto_tol": float(recipe.get("bg_veto_tol", bgmodel.VM_TOL)),
                     "centers": [[round(float(v), 2) for v in c] for c in bg_centers]},
        "keep_region_blocks": keep_region_blocks,
        "expect_continuity": bool(case.thin) or path.size > 0,
    }
    card = score.evaluate(case, src, src, alpha, ctx, thresholds)
    # 包级适配：只记录，不参与判定（细结构覆盖率已经作为 continuity 的输入用过）。
    # 包里没有 runs/ 与报告层，selfcheck 靠这一行打印一份可读数值。
    card["metrics"]["thin_cov"] = None if thin_cov is None else round(float(thin_cov), 4)
    if unresolved:
        card["unresolved_keep"] = unresolved
        card["failures"].append({"metric": "unresolved_keep", "value": unresolved,
                                 "limit": "keep 的每一项都要在原图里定位到",
                                 "attribution": "P1/P2：提案层没找到该物体（原图里可能不存在）"})
        card["status"] = "uncertain"
        card["blocked_layer"] = "P1/P2"
        card["score"] = round(min(card["score"], 5.0), 4)   # 有"定位不到"的 keep 项，不许排在及格配方前面

    rgba = np.dstack([src, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)])
    result = {"case": case.id, "card": card, "rgba": rgba, "alpha": alpha,
              "proposals": ps, "rod_line": rod_line, "shafts": shafts}

    if write:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        imaging.save_rgba(od / f"{case.id}.rgba.png", rgba)
        imaging.save_gray(od / f"{case.id}.mask.png", (alpha * 255).astype(np.uint8))
        imaging.save_gray(od / f"{case.id}.estimated.png", (already_est.astype(np.uint8) * 255))
        (od / f"{case.id}.scorecard.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        recipes.save(recipe, od / f"{case.id}.recipe.json")
        _proof(src, alpha, od / f"{case.id}.proof.png")
        result["out_dir"] = str(od)
    return result
