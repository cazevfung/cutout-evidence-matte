---
name: cutout-evidence-matte
description: 抠图、去背景、要透明底、发丝或半透明边缘、细长物体抠不干净时使用。用三路候选提案（显著性去背景、图层拆分、绿幕对齐）加本地几何重建与颜色线 matting 产出 RGBA：像素只取自原图，边界 alpha 本地估算，最后用六指标和三条不依赖提案分母的自证给出 pass/uncertain。不用于换背景合成、超分、视频抠像。
metadata:
  short-description: 证据驱动抠图：像素来自原图，边界本地 matting，过不了就说缺哪一层
---

# 证据驱动抠图

这套方法解决的是「生成模型给的图看着像、但边缘和像素不能用」的问题。分工固定：

- 语义模型只回答「图里有什么、哪些不要」，产出**候选 mask**；
- 最终像素只从原图取；
- 边界 alpha 在本地按颜色线估，不依赖云端 matting 端点；
- 过不了判据就报 `uncertain`，并写明缺的是哪一层。

下面所有数字都是冻结的判据，不是建议值。完整阈值表、归因表与 uncertain 口径见
[`docs/thresholds.md`](docs/thresholds.md)。

## 什么时候用 / 不用

用：用户要抠图、去背景、透明底 PNG；主体有发丝、纱、玻璃、水花等半透明边缘；
鱼线、竿梢、细绳、天线这类细长物体抠不干净或断成几截；需要「这张到底过没过」的数值判据。

不用：只要换背景合成（用普通图像编辑）、超分、视频抠像、批量分发面板。

## 一条命令

```bash
# 先校验采集缓存（P1/P2/P3 齐不齐、bbox 在不在原图范围）
python scripts/cutout_run.py check-cache --case <case-dir> --cache-dir <cache-root>

# 跑决策层，产出 rgba / mask / scorecard / recipe / proof
python scripts/cutout_run.py run --case <case-dir> --cache-dir <cache-root> --out out

# 完全离线自检：一张原图 + 三份手写产物，打印六指标与 pass
python scripts/selfcheck.py
```

`case-dir` 里放 `case.json` 与 `source.*`；`case.json` 至少写：

```json
{"id": "my-case", "source": "source.png",
 "keep": ["角色", "细线"], "exclude": ["水花"], "thin": ["细线"]}
```

`keep` 是必须出现在结果里的东西，`exclude` 是必须压住的东西，`thin` 是细结构清单。
识别不了的 `keep` 项不会被静默放过——它会进 `unresolved_keep`，判 `uncertain`。

## 采集层：三条产物的最小契约

决策层只认目录，不关心这些文件是自动化采集还是人手放的。目录形状：

```
<cache-root>/<case_id>/<tag>/
  meta.json
  bria.png              # P1 显著性去背景：RGBA，alpha 即候选 mask
  key.png               # P3 绿幕编辑：RGB，主体画在纯 #00FF00 上
  layers/z1_xxx.png ... # P2 图层拆分：RGBA，每层一张
```

`meta.json` 只需这些字段：

```json
{
  "case_id": "my-case",
  "bria": "bria.png",
  "key": "key.png",
  "layers": [
    {"z": 0, "name": "background", "label": "背景",  "bbox": [0, 0, 1920, 1080], "file": "layers/z0_background.png"},
    {"z": 1, "name": "character",  "label": "角色",  "bbox": [420, 180, 1180, 980], "file": "layers/z1_character.png"},
    {"z": 2, "name": "splash",     "label": "水花",  "bbox": [1500, 700, 1900, 1000], "file": "layers/z2_splash.png"}
  ]
}
```

规则只有四条：

1. **bbox 是原图坐标** `[x0, y0, x1, y1]`，不是图层图自己的坐标。图层 PNG 可以是 bbox 大小，
   加载时按 bbox 放回原图；不要缩放原图去迁就图层。
2. **P1 只放最显眼的主体**（通常是人物）。它不许包含「不要」的东西。
3. **P2 的 `label` 写明这一层是什么**。`classify()` 按中文关键词分三类：角色/人物 → keep 主体；
   水花/光斑/特效 → 排除；背景/道具/房屋 → 非 keep（融合时做减法）。分不出类的层按
   「其它」处理，同样进非 keep 集合。
4. **P3 是绿幕编辑图**，16:9 之类的不同画幅没关系，加载时会按尺度搜索对齐回原图；
   **对齐误差 >4px 该路作废**，只在 P1/P2 漏主体时补位。

人工落缓存就三步：建目录 → 放三个文件 → 写 `meta.json`。放完先跑 `check-cache`；
缺哪一路会直接点名（P1 / P2 / P3 / meta.json / 某张层图），不会给一个裸栈。
`demo/` 就是一份可以直接照抄的手写样例。

## 决策层：五条不能动的规则

### 1. 像素只能来自原图

最终 RGBA 的 RGB 对每个 `alpha>0` 的像素与原图**同坐标逐通道相等**，`pixel_fidelity` 必须是 0。
生成模型、超分、重采样都不许碰最终像素；P1/P2/P3 只提供候选 mask。
唯一例外是细结构几何重建像素，必须逐块标 `estimated` 并写理由。

### 2. 三路提案，融合做减法

- P1 显著性去背景：给最显眼主体，常把背景/大块环境也当主体；
- P2 图层拆分：人/道具/背景分层，用来**否决** P1 把背景当主体；
- P3 绿幕编辑：对齐回原图后只在 P1/P2 漏主体时补位。

融合顺序：P1 为骨架 → P2 只补与 P1 相邻（默认 3px）且面积达标的连通块 → P3 只在整体缺失时补位。
**减法规则**：某像素被 P2 明确归入非 keep 层、且该像素不落在任何 keep 层（外扩 1px 保护带）里，
就从融合结果剔除。P1 只在 P2 没意见的地方保留主导权。这条不看题材。
面积 < 图面积 0.05% 的碎片丢弃；与主体不连通的半透明块按连通性过滤。
每个被丢的块都要记进 `dropped`，不许悄悄扔。

### 3. 细结构不交给分割模型

判据：**长宽比 >12 且局部宽度 <3px**。这类目标（鱼线、竿梢、细绳、天线）语义分割拿不到，
只允许从原图几何重建：top-hat / black-hat 抬细线 → HoughLinesP 出线段 → 共线合并 → 端点外推。

两条通用门槛，不看题材：

- **邻接证据**：至少一端真的落在 keep 主体里（主体外扩 2px 内）。「离主体 60px 以内」不算。
- **不是纯背景**：线段沿线 ≥80% 落在非 keep 层里就丢弃。

重建出 1.5–2.0px 宽、带 0.5px 软边的 alpha 带；整条通道的像素标 `estimated`，
scorecard 里单列「重建了哪条细结构、多少像素」。

### 4. 边界 alpha 本地估

由二值 mask 生成 trimap（默认腐蚀 2px、膨胀 1–3px），只在未知带里解颜色线：

```
alpha ≈ (I - B)·(F - B) / |F - B|²
```

F、B 用局部盒滤波从「已知前景 / 已知背景」估，纯 numpy/OpenCV，全分辨率，不调云端。
已知前景 alpha=1、已知背景 alpha=0。`|F-B|` 太小（前景背景同色）时该像素标 `low_confidence`，
退回距离场几何 alpha，并计入 `hole` / `halo` 的来源统计——不许静默当成功。

### 5. 填洞先看颜色

轮廓内的洞不无条件填。用 Lab ΔE 判断洞里的像素像不像背景色（背景色模型从 P2 非 keep 层 +
图像四边框带取色，排除 keep 外扩 2px）：

- 像背景（ΔE ≤5.0）→ **保留这个洞**，那是围出来的真实缝隙；
- 不像（例如角色自己的闭合缝）→ 填；
- 面积 <200px 的小洞一律照填，免得把噪点留成针孔。

同理，alpha>0 但颜色几乎就是背景、又不在轮廓 2px 软边带里的成片像素（≥800px 且同质）
会被否决掉——这条用来压水体/地面漏进来的环境色。

## 判据：六指标 + 三条自证

主指标（全部要过）：

| 指标 | 阈值 | 抓什么 |
| --- | --- | --- |
| halo | ≤ 30.0 | 轮廓外 3px 环带在洋红/近白底上的颜色溢出 |
| stray | ≤ 12 个连通域 | 轮廓外的可见碎点 |
| hole | ≤ 0.01 | 轮廓内露出背景色的孔洞占比 |
| coverage | ≥ 0.8 | `keep` 每一项被 alpha 覆盖的最小比例 |
| continuity | ≥ 0.95 | 竿顶→竿梢→线尾采样路径上 alpha≥0.5 的比例 |
| pixel_fidelity | ≤ 0 | `alpha>0` 像素与原图的最大通道差（恒为 0） |

参考分（不作唯一判据）：`iou_ref ≥0.9`、`ref_cut ≤0.02`（参照若来自另一机位，
IoU 判 1px 容差、裁切判 2px 容差，0px 严格值只记录）；`exclude_hit ≤0.05`；`thin_cov ≥0.95`。

三条**与提案分母无关**的自证，专门抓「P1 把背景当主体」：

- `alpha_area_frac ≤0.5`：前景占全图比例；
- `alpha_evidence_ratio ≤2.0`：前景面积 ÷ P2 keep 层证据面积；
- `p1_leak`：P1 占比 >0.3 **且** `alpha_evidence_ratio >1.5` 才判红——只是 P1 大不算错，
  被 P2 否决掉就没事。

**过不了就是 `uncertain`**，不是「差一点」。scorecard 必须写清 `failures`（指标、数值、限值）
与 `blocked_layer`，取值只能是 P1 / P2 / P3 / 几何 / alpha 中的一个或多个；
进程仍退出 0，但交付物里带 `failures` 与 `unresolved_keep`，禁止静默交一张错的图。
三条口径（U1/U2/U3）在 [`docs/thresholds.md`](docs/thresholds.md)。

## 依赖与失败模式

运行依赖只有两个：

```bash
pip install -r requirements.txt     # numpy, opencv-python
```

`scipy`、`Pillow` 不是运行依赖（代码里不 import）。缺库时脚本会直接说缺哪个、装什么，
不会给 ImportError 栈。

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| `缺 P1（显著性提案）` | 缓存缺 `bria.png` | 补 P1，或检查 `meta.json` 的 `bria` 字段 |
| `缺 P2 图层文件` / `缺 P2` | `layers/` 空或 meta 的 layers 为空 | 至少给一层；顶层要保留的主体写成 label 含「角色」 |
| `缺 P3（绿幕提案）` | 缺 `key.png` | 补绿幕图；主配方默认不用，但契约要求三路齐备 |
| `P2 图层 bbox 无效` | bbox 不是原图坐标或为空 | 按 `[x0,y0,x1,y1]` 重填 |
| `unresolved_keep` | keep 里的东西原图里根本没定位到 | 图里确实没有就改 keep；有但漏了按归因补 P1/P2 |
| 输出 `uncertain` + `blocked_layer` | 判据没过 | 按归因表动那一层，不许调阈值绕过 |
| 输出糊/断边 | 走了生成式重绘或超分 | 最终像素必须回到原图，候选 mask 才可以用模型 |

## 不做什么

- 不改判据迎合单张图：阈值只能由重新标定改动，不能为了让某图过而放宽。
- 不把生成模型输出的像素当结果，不用它补画被遮挡的细结构。
- 不做 GUI、不做批量分发、不做视频。
- 不把手写 demo 当唯一通路：真实案例同样走 `run --case ... --cache-dir ...`。
