"""R7 六指标 + 归因。评分器先能判红，再说好不好。

指标：halo / stray / hole / coverage / continuity / pixel_fidelity，外加参考分 iou_ref。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imaging

DEFAULT_THRESHOLDS = {
    "halo_max": 30.0,          # 标定自人工参照自身的 27.64（见 docs/thresholds.md）
    "stray_max": 12,
    "hole_max": 0.01,
    "coverage_min": 0.80,
    "continuity_min": 0.95,
    "iou_ref_min": 0.90,
    "ref_cut_max": 0.02,
    "exclude_hit_max": 0.05,
    "alpha_area_frac_max": 0.50,       # 前景占全图比例上限（与提案无关的形态检查）
    "alpha_evidence_ratio_max": 2.0,   # alpha 面积 / keep 证据面积
    "p1_leak_ratio_max": 1.5,          # P1 面积异常且真的漏进结果时判红
    "p1_area_frac_warn": 0.30,
    "fidelity_max": 0,
    "thin_cov_min": 0.95,
}

ATTRIBUTION = {
    "halo": "alpha：边界去污染 / trimap 半径",
    "stray": "P1/P2：最小面积与连通性过滤",
    "hole": "alpha：内部孔洞与低对比区",
    "coverage": "P1/P2：提案漏了 keep 清单里的东西",
    "continuity": "几何：细结构没重建出来",
    "pixel_fidelity": "像素来源：生成图或重采样混进了结果，RGB 必须回到原图",
    "iou_ref": "整体形状：融合或排除区判错",
    "exclude_hit": "P1/P2：排除区没压住",
    "alpha_area_frac": "P1/P2：前景面积占全图过大，说明背景被当成了主体",
    "alpha_evidence_ratio": "P1/P2：输出面积远超 keep 层证据面积",
    "p1_leak": "P1/P2：显著性提案面积异常，且异常部分漏进了结果",
    "thin_cov": "几何：细结构覆盖不足",
}


def halo_metric(rgb: np.ndarray, alpha: np.ndarray) -> float:
    solid = alpha > 0.5
    ring = imaging.dilate(solid, 3) & ~solid
    if not ring.any():
        return 0.0
    errs = []
    for bg in ((255, 0, 255), (250, 250, 250)):
        comp = imaging.composite(rgb, alpha, bg).astype(np.float32)
        bgv = np.array(bg, np.float32).reshape(1, 3)
        errs.append(float(np.linalg.norm(comp[ring] - bgv, axis=1).mean()))
    return float(np.mean(errs))


def stray_metric(rgb: np.ndarray, alpha: np.ndarray) -> int:
    """轮廓外的深色高频碎点：数"贴白底看得见的、与主体不相连的小连通域"。

    判据分两步：先找 alpha 上离主体远的孤立块（轮廓外），再看它合成到白底后是否够深。
    这样既不数主体内部纹理，也不会把有意保留的分离物体（鱼，面积远大于 60）算进去。
    """
    solid = alpha > 0.5
    if not solid.any():
        return 0
    n, lab, stats, _ = imaging.label(solid)
    main = lab == (1 + int(np.argmax(stats[1:, 4])))
    comp = imaging.composite(rgb, alpha, (255, 255, 255))
    lum = imaging.luminance(comp)
    far = (alpha > 0.05) & ~imaging.dilate(main, 3)
    n2, lab2, stats2, _ = imaging.label(far)
    count = 0
    for i in range(1, n2):
        area = int(stats2[i, 4])
        if not (1 <= area <= 400):
            continue
        comp_px = lab2 == i
        if float(alpha[comp_px].sum()) < 3.0:
            continue          # alpha 太薄，贴什么底都看不见，不算碎点
        count += 1
    return count


def hole_metric(rgb: np.ndarray, alpha: np.ndarray) -> float:
    solid = alpha > 0.5
    inside = imaging.erode(solid, 2)
    if not inside.any():
        return 0.0
    cm = imaging.composite(rgb, alpha, (255, 0, 255)).astype(np.float32)
    cw = imaging.composite(rgb, alpha, (250, 250, 250)).astype(np.float32)
    is_mag = np.linalg.norm(cm - np.array([255, 0, 255], np.float32), axis=2) < 18
    is_wht = np.linalg.norm(cw - np.array([250, 250, 250], np.float32), axis=2) < 18
    hole = inside & is_mag & is_wht
    return float(hole.sum() / max(inside.sum(), 1))


def fidelity_metric(rgb: np.ndarray, src_rgb: np.ndarray, alpha: np.ndarray) -> tuple[int, int]:
    m = alpha > 0
    if not m.any():
        return 0, 0
    d = np.abs(rgb[m].astype(np.int16) - src_rgb[m].astype(np.int16))
    return int(d.max()), int((d.max(axis=1) > 0).sum())


def region_overlap(region: np.ndarray, solid: np.ndarray) -> float:
    if region.sum() == 0:
        return 1.0
    return float((region & solid).sum() / region.sum())


def continuity_metric(alpha: np.ndarray, path: np.ndarray) -> dict:
    from . import thin

    if path.size == 0:
        return {"coverage": 0.0, "connected": False, "n": 0}
    cov = thin.coverage(alpha, path, thr=0.5, tol=1)
    hard = alpha > 0.35
    n, lab, stats, _ = imaging.label(hard)

    def lab_near(p, rad=6):
        """端点附近（rad 内）最近的有标签像素的连通域编号。"""
        x = int(round(p[0])); y = int(round(p[1]))
        best = (10 ** 9, 0)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < hard.shape[0] and 0 <= xx < hard.shape[1] and lab[yy, xx]:
                    d2 = dx * dx + dy * dy
                    if d2 < best[0]:
                        best = (d2, int(lab[yy, xx]))
        return best[1]

    l0 = lab_near(path[0]) if len(path) else 0
    l1 = lab_near(path[-1]) if len(path) else 0
    connected = l0 != 0 and l0 == l1
    return {"coverage": float(cov), "connected": bool(connected), "n": int(len(path))}


def _sift_matches(src_rgb: np.ndarray, ref_rgba: np.ndarray):
    g1 = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(ref_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None, None
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None, None
    return (np.float32([k1[m.queryIdx].pt for m in good]),
            np.float32([k2[m.trainIdx].pt for m in good]))


def _kmeans(points: np.ndarray, k: int, iters: int = 30, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(points))
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    labels = np.zeros(len(points), np.int32)
    for _ in range(iters):
        d = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = points[labels == j].mean(axis=0)
    return labels


def _sane(H: np.ndarray) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
    persp = abs(H[2, 0]) + abs(H[2, 1])
    return 0.2 < scale < 5.0 and persp < 0.01


def piecewise_models(src_rgb: np.ndarray, ref_rgba: np.ndarray, k: int = 4,
                     min_inliers: int = 25, sigma: float = 90.0) -> list[dict]:
    p1, p2 = _sift_matches(src_rgb, ref_rgba)
    if p1 is None:
        return []
    labels = _kmeans(p1, k)
    models = []
    for j in range(int(labels.max()) + 1):
        sel = labels == j
        if sel.sum() < 12:
            continue
        H, mask = cv2.findHomography(p2[sel], p1[sel], cv2.RANSAC, 2.5)
        if H is None or mask is None or not _sane(H):
            continue
        inl = int(mask.sum())
        if inl < min_inliers:
            continue
        pts = p1[sel][mask.ravel() > 0]
        models.append({"H": H, "pts": pts, "inliers": inl, "points": int(sel.sum()),
                       "sigma": sigma})
    return models


def warp_ref_piecewise(ref_mask: np.ndarray, shape: tuple[int, int], models: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    if not models:
        return np.zeros(shape, bool), np.zeros(shape, bool)
    # 用各簇内点的核密度决定"这块归哪个单应管"，比按簇中心划分更贴近证据分布
    dens = np.zeros((len(models), h, w), np.float32)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    for i, m in enumerate(models):
        sig = float(m.get("sigma", 90.0))
        for (px, py) in m["pts"]:
            d2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
            dens[i] += np.exp(-d2 / (2 * sig * sig)).astype(np.float32)
    near = dens.argmax(axis=0)
    out = np.zeros(shape, bool)
    valid = np.zeros(shape, bool)
    rm = ref_mask.astype(np.uint8)
    vv = np.ones(ref_mask.shape, np.uint8)
    for i, m in enumerate(models):
        warped = cv2.warpPerspective(rm, m["H"], (w, h), flags=cv2.INTER_NEAREST) > 0
        wvalid = cv2.warpPerspective(vv, m["H"], (w, h), flags=cv2.INTER_NEAREST) > 0
        sel = near == i
        out |= warped & sel
        valid |= wvalid & sel
    return out, valid


def _subject_registration(src_rgb: np.ndarray, ref_rgba: np.ndarray, solid: np.ndarray,
                          radius: int = 40) -> np.ndarray | None:
    """只用落在"主体核"附近的特征点拟合单应。

    参照是另一机位（见 runs/probe/probe-01.md）：整幅场景的特征点里混着背景的视差，
    用全场景点位会把主体上的对齐拖歪。既然要比起的就是前景形状，就用前景附近的点位定标。
    """
    p1, p2 = _sift_matches(src_rgb, ref_rgba)
    if p1 is None:
        return None
    # 只保留"实心主体"：腐蚀 6px 会把 1–2px 的鱼线、竿梢削掉，剩下最大的一块就是角色的躯干
    bod = imaging.erode(solid, 6)
    if not bod.any():
        bod = solid
    n, lab, stats, _ = imaging.label(bod)
    biggest = lab == (1 + int(np.argmax(stats[1:, 4])))
    core = imaging.dilate(biggest, radius)
    keep = np.array([core[min(max(int(round(y)), 0), core.shape[0] - 1),
                           min(max(int(round(x)), 0), core.shape[1] - 1)] for x, y in p1])
    if keep.sum() < 12:
        return None
    H, mask = cv2.findHomography(p2[keep], p1[keep], cv2.RANSAC, 2.5)
    if H is None or mask is None or int(mask.sum()) < 12 or not _sane(H):
        return None
    return H


def _iou_against(refA: np.ndarray, solid: np.ndarray, tol: int = 0) -> tuple[float, float]:
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        a = cv2.dilate(solid.astype(np.uint8), k) > 0
        b = cv2.dilate(refA.astype(np.uint8), k) > 0
    else:
        a, b = solid, refA
    i = (a & b).sum()
    u = (a | b).sum()
    cut = (refA & ~a).sum() / max(refA.sum(), 1)
    return float(i / max(u, 1)), float(cut)


def iou_ref_metric(src_rgb: np.ndarray, solid: np.ndarray, ref_rgba: np.ndarray,
                   homography: np.ndarray | None) -> dict:
    if ref_rgba is None:
        return {"iou": None, "cut": None, "note": "no reference"}
    h, w = solid.shape
    ref_mask = ref_rgba[..., 3] > 128
    vv = np.ones(ref_rgba.shape[:2], np.uint8)
    H = homography if homography is not None else estimate_homography(src_rgb, ref_rgba)
    if H is None:
        return {"iou": None, "cut": None, "note": "registration failed"}
    refA = cv2.warpPerspective(ref_mask.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    if not refA.any():
        return {"iou": None, "cut": None, "note": "empty reference after warp"}
    iou_g, cut_g = _iou_against(refA, solid, 0)
    iou_g1, _ = _iou_against(refA, solid, 1)
    iou_g2, cut_g2 = _iou_against(refA, solid, 2)

    out = {"iou": iou_g, "cut": cut_g, "iou_tol1": iou_g1, "cut_tol2": cut_g2,
           "iou_global": iou_g, "cut_global": cut_g, "mode": "global_homography"}

    Hs = _subject_registration(src_rgb, ref_rgba, solid)
    if Hs is not None:
        refS = cv2.warpPerspective(ref_mask.astype(np.uint8), Hs, (w, h), flags=cv2.INTER_NEAREST) > 0
        if refS.any():
            iou_s, cut_s = _iou_against(refS, solid, 0)
            iou_s1, _ = _iou_against(refS, solid, 1)
            iou_s2, cut_s2 = _iou_against(refS, solid, 2)
            out.update({"iou": iou_s, "cut": cut_s, "iou_tol1": iou_s1, "cut_tol2": cut_s2,
                        "mode": "subject_registration"})
    return out


def estimate_homography(src_rgb: np.ndarray, ref_rgba: np.ndarray) -> np.ndarray | None:
    g1 = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(ref_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 3.0)
    return H


def evaluate(case, src_rgb: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, ctx: dict,
             thresholds: dict | None = None) -> dict:
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    solid = alpha > 0.5

    fid_max, fid_n = fidelity_metric(rgb, src_rgb, alpha)
    metrics = {
        "halo": round(halo_metric(rgb, alpha), 3),
        "stray": stray_metric(rgb, alpha),
        "hole": round(hole_metric(rgb, alpha), 5),
        "pixel_fidelity": fid_max,
        "fidelity_pixels": fid_n,
    }
    failures = []

    cov_detail = {}
    region_masks = ctx.get("keep_regions", {})
    worst = 1.0
    for name, region in region_masks.items():
        ov = region_overlap(region, solid)
        cov_detail[name] = round(ov, 4)
        worst = min(worst, ov)
    metrics["coverage"] = round(worst, 4)

    ex_hit = {}
    for name, region in ctx.get("exclude_regions", {}).items():
        ex_hit[name] = round(float((region & solid).sum() / max(region.sum(), 1)), 4)
    metrics["exclude_hit"] = max(ex_hit.values()) if ex_hit else 0.0

    path = ctx.get("path", np.zeros((0, 2)))
    expect_cont = ctx.get("expect_continuity", path.size > 0)
    if path.size == 0:
        cont = {"coverage": None, "connected": None, "n": 0}
        metrics["continuity"] = None
        metrics["continuity_connected"] = None
        metrics["continuity_note"] = "本题材没有细结构（thin 为空且没检出竿/线），不判"
    else:
        cont = continuity_metric(alpha, path)
        metrics["continuity"] = round(cont["coverage"], 4)
        metrics["continuity_connected"] = cont["connected"]

    iou = iou_ref_metric(src_rgb, solid, ctx.get("reference"), ctx.get("homography"))
    metrics["iou_ref"] = None if iou["iou"] is None else round(iou["iou"], 4)          # 0px 严格
    metrics["ref_cut"] = None if iou["cut"] is None else round(iou["cut"], 4)          # 0px 严格
    metrics["iou_ref_tol1"] = None if iou.get("iou_tol1") is None else round(iou["iou_tol1"], 4)
    metrics["ref_cut_tol2"] = None if iou.get("cut_tol2") is None else round(iou["cut_tol2"], 4)
    metrics["iou_ref_global"] = None if iou.get("iou_global") is None else round(iou["iou_global"], 4)
    metrics["registration"] = iou.get("mode")
    if iou.get("note"):
        metrics["iou_ref_note"] = iou["note"]

    if metrics["halo"] > th["halo_max"]:
        failures.append({"metric": "halo", "value": metrics["halo"], "limit": th["halo_max"],
                         "attribution": ATTRIBUTION["halo"]})
    if metrics["stray"] > th["stray_max"]:
        failures.append({"metric": "stray", "value": metrics["stray"], "limit": th["stray_max"],
                         "attribution": ATTRIBUTION["stray"]})
    if metrics["hole"] > th["hole_max"]:
        failures.append({"metric": "hole", "value": metrics["hole"], "limit": th["hole_max"],
                         "attribution": ATTRIBUTION["hole"]})
    if metrics["coverage"] < th["coverage_min"]:
        failures.append({"metric": "coverage", "value": metrics["coverage"], "limit": th["coverage_min"],
                         "attribution": ATTRIBUTION["coverage"]})
    if expect_cont and (metrics["continuity"] is None
                        or metrics["continuity"] < th["continuity_min"]
                        or not cont["connected"]):
        failures.append({"metric": "continuity", "value": metrics["continuity"], "limit": th["continuity_min"],
                         "connected": cont["connected"], "attribution": ATTRIBUTION["continuity"]})
    if metrics["pixel_fidelity"] > th["fidelity_max"]:
        failures.append({"metric": "pixel_fidelity", "value": metrics["pixel_fidelity"],
                         "limit": th["fidelity_max"], "attribution": ATTRIBUTION["pixel_fidelity"]})
    # AC07 判定口径：参照来自另一机位，边界自身就有 1–2px 抖动，
    # 所以 IoU 判 1px 容差口径、被裁比例判 2px 容差口径；0px 严格值照样记在 metrics 里。
    iou_judged = metrics.get("iou_ref_tol1", metrics["iou_ref"])
    cut_judged = metrics.get("ref_cut_tol2", metrics["ref_cut"])
    if iou_judged is not None and iou_judged < th["iou_ref_min"]:
        failures.append({"metric": "iou_ref", "value": iou_judged, "limit": th["iou_ref_min"],
                         "protocol": "iou@1px", "attribution": ATTRIBUTION["iou_ref"]})
    if cut_judged is not None and cut_judged > th["ref_cut_max"]:
        failures.append({"metric": "ref_cut", "value": cut_judged, "limit": th["ref_cut_max"],
                         "protocol": "cut@2px", "attribution": ATTRIBUTION["iou_ref"]})
    if metrics["exclude_hit"] > th["exclude_hit_max"]:
        failures.append({"metric": "exclude_hit", "value": metrics["exclude_hit"], "limit": th["exclude_hit_max"],
                         "attribution": ATTRIBUTION["exclude_hit"]})
    # ---- 返工3：不依赖提案分母的自证检查 ----
    keep_px = int(ctx.get("keep_pixels", 0))
    p1_px = int(ctx.get("p1_pixels", 0))
    total_px = int(solid.size)
    metrics["alpha_area_frac"] = round(float(solid.sum() / max(total_px, 1)), 4)
    metrics["alpha_evidence_ratio"] = (round(float(solid.sum() / keep_px), 3) if keep_px else None)
    metrics["p1_area_frac"] = round(float(p1_px / max(total_px, 1)), 4)
    nonkeep = np.zeros(solid.shape, bool)
    for m in ctx.get("nonkeep_layers", []):
        nonkeep |= m
    if nonkeep.any():
        probe_den = max(int((nonkeep & ~ctx.get("keep_mask", np.zeros_like(nonkeep))).sum()), 1)
        metrics["exclude_probe_hit"] = round(
            float((solid & nonkeep & ~ctx.get("keep_mask", np.zeros_like(nonkeep))).sum() / probe_den), 4)

    if metrics["alpha_area_frac"] > th["alpha_area_frac_max"]:
        failures.append({"metric": "alpha_area_frac", "value": metrics["alpha_area_frac"],
                         "limit": th["alpha_area_frac_max"], "attribution": ATTRIBUTION["alpha_area_frac"]})
    if (metrics["alpha_evidence_ratio"] is not None
            and metrics["alpha_evidence_ratio"] > th["alpha_evidence_ratio_max"]):
        failures.append({"metric": "alpha_evidence_ratio", "value": metrics["alpha_evidence_ratio"],
                         "limit": th["alpha_evidence_ratio_max"],
                         "attribution": ATTRIBUTION["alpha_evidence_ratio"]})
    if (metrics["p1_area_frac"] > th["p1_area_frac_warn"]
            and (metrics["alpha_evidence_ratio"] or 0) > th["p1_leak_ratio_max"]):
        failures.append({"metric": "p1_leak",
                         "value": {"p1_area_frac": metrics["p1_area_frac"],
                                   "alpha_evidence_ratio": metrics["alpha_evidence_ratio"]},
                         "limit": {"p1_area_frac_warn": th["p1_area_frac_warn"],
                                   "alpha_evidence_ratio_max": th["p1_leak_ratio_max"]},
                         "attribution": ATTRIBUTION["p1_leak"]})

    thin_cov = ctx.get("thin_coverage")
    if thin_cov is not None and thin_cov < th["thin_cov_min"]:
        failures.append({"metric": "thin_cov", "value": round(thin_cov, 4), "limit": th["thin_cov_min"],
                         "attribution": ATTRIBUTION["thin_cov"]})

    layer = None
    if failures:
        layer = sorted({f["attribution"].split("：")[0] for f in failures}).pop(0)

    # 先看"过没过"，再看"过得漂不漂亮"：任何 pass 的配方都必须压过任何 uncertain 的配方
    quality = 0.0
    quality += 0.40 * float(iou_judged if iou_judged is not None else 0.0)
    quality += 0.20 * (1.0 - min(metrics["halo"] / max(th["halo_max"], 1e-6), 1.0))
    quality += 0.20 * float(metrics["coverage"] or 0.0)
    cont_v = metrics["continuity"]
    quality += 0.20 * float(1.0 if cont_v is None else cont_v)
    score = (10.0 if not failures else 0.0) + quality

    card = {
        "case_id": case.id if hasattr(case, "id") else str(case),
        "status": "uncertain" if failures else "pass",
        "metrics": metrics,
        "thresholds": th,
        "coverage_detail": cov_detail,
        "exclude_detail": ex_hit,
        "failures": failures,
        "blocked_layer": layer,
        "score": round(float(score), 4),
        "estimated": ctx.get("estimated", []),
        "keep_region_blocks": ctx.get("keep_region_blocks", []),
        "bg_model": ctx.get("bg_model", {}),
        "hole_decisions": ctx.get("hole_decisions", []),
        "bg_veto": ctx.get("bg_veto", []),
        "low_confidence_pixels": int(ctx.get("low_conf_pixels", 0)),
        "dropped": ctx.get("dropped", []),
        "recipe": ctx.get("recipe", {}),
        "cache_tag": ctx.get("cache_tag", ""),
        "capture_calls": ctx.get("capture_calls", {}),
        "thin_structures": ctx.get("thin_structures", []),
    }
    return card
