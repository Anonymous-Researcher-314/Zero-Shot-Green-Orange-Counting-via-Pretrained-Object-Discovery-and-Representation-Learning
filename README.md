# Zero-Shot Green Orange Counting via Pretrained Object Discovery and Representation Learning

> **Systematic benchmark of pretrained unsupervised models for counting green, unripe oranges from 4K orchard video — evaluated zero-shot across 10,577 annotated frames.**

---

## Overview

This repository contains the annotation files, inference notebooks, and evaluation code for the paper:

**"Self-Supervised Green Orange Counting via Object Discovery and Representation Learning"**

We evaluate eight pretrained unsupervised models for counting green (immature) oranges from 4K orchard video in a strictly zero-shot setting — no model weights are modified, and no orange imagery of any kind is used during model development. Ground truth annotations are accessed only at evaluation time to compute metrics.

> **Dataset availability:** Annotation files, inference code, and evaluation notebooks are available in this repository. The full frame dataset (10,577 × 4K UHD frames) will be made publicly available upon camera-ready submission.

---

## Key Results

| Model | Strategy | MAE ↓ | RMSE ↓ | W±2% ↑ | Bias | F1 ↑ | Time |
|-------|----------|--------|--------|--------|------|------|------|
| **DINOv2** | Full-image k-means | **5.28** | **6.67** | **30.2%** | −0.64 | 0.142 | ~88 min |
| CutLER | Tiled 8×1920×1080 | 5.97 | 7.55 | 26.4% | −2.15 | 0.179 | 118 min |
| MAE | Full-image k-means | 6.05 | 7.40 | 23.4% | −1.10 | 0.093 | 29 min |
| iBOT | Full-image k-means | 6.21 | 7.67 | 23.3% | −3.33 | 0.099 | 28 min |
| CuVLER | Tiled 8×1920×1080 | 6.61 | 8.12 | 22.9% | +1.92 | 0.156 | 132 min |
| DINO | Full-image attention | 6.73 | 8.17 | 21.4% | −5.97 | 0.137 | ~14 min |
| DETReg | Full-image resize | 8.11 | 9.97 | 18.3% | +4.58 | **0.406** | 80 min |
| LOST | Sliding window | 9.92 | 11.01 | 2.6% | −9.92 | 0.005 | ~176 min |

*Test split: Tree 02, Tree 03, Tree 09, Tree 10 — 4,210 frames. GT mean: 11.71 oranges/frame.*
*All experiments run on NVIDIA A100 80GB PCIe, PyTorch 2.4.0, CUDA 12.1, Python 3.11.9.*

---

## Dataset

### Overview

| Property | Value |
|----------|-------|
| Total frames | 10,577 |
| Resolution | 3840 × 2160 (4K UHD, portrait) |
| Trees | 10 orange trees |
| Videos | 30 (3 per tree) |
| Groups | Group 1: 5 trees × 30-second videos; Group 2: 5 trees × 40-second videos |
| Frame extraction | 10 FPS |
| GT annotation type | Manual point annotations (dot per orange) |
| Total GT dots | 123,254 hand-placed points |
| GT range | 0 – 45 oranges per frame |
| GT mean | 11.71 oranges per frame |
| GT std | 6.43 |
| Orchard | Commercial orange orchard, semi-arid Mediterranean climate region |
| Varieties | Valencia and Navel (approximately 400 trees, 5-acre site) |
| Recording device | iPhone camera, .mov format, 4K UHD |
| Recording time | Midday and afternoon sessions |

### Per-Tree Statistics

| Tree ID | Group | Vid 01 | Vid 02 | Vid 03 | Total | GT Mean | Split |
|---------|-------|--------|--------|--------|-------|---------|-------|
| Tree 01 | 30-second | 301 | 305 | 300 | 906 | 16.45 | CAL |
| Tree 02 | 30-second | 300 | 300 | 300 | 900 | 12.75 | **TEST** |
| Tree 03 | 30-second | 300 | 306 | 300 | 906 | 10.08 | **TEST** |
| Tree 04 | 30-second | 300 | 315 | 316 | 931 | 15.22 | CAL |
| Tree 05 | 30-second | 305 | 305 | 300 | 910 | 11.05 | CAL |
| Tree 06 | 40-second | 400 | 403 | 400 | 1,203 | 10.57 | CAL |
| Tree 07 | 40-second | 403 | 405 | 400 | 1,208 | 19.74 | CAL |
| Tree 08 | 40-second | 404 | 400 | 405 | 1,209 | 5.03 | CAL |
| Tree 09 | 40-second | 400 | 400 | 404 | 1,204 | 6.03 | **TEST** |
| Tree 10 | 40-second | 400 | 400 | 400 | 1,200 | 11.82 | **TEST** |
| **Total** | | | | | **10,577** | **11.71** | |

### Calibration / Test Split

| Split | Trees | Frames | Purpose |
|-------|-------|--------|---------|
| Calibration | Tree 01, 04, 05, 06, 07, 08 | 6,367 | Parameter selection only (54 sweep frames used) |
| **Test** | **Tree 02, 03, 09, 10** | **4,210** | **Primary evaluation — never seen during parameter selection** |
| Full dataset | All 10 trees | 10,577 | Supplementary result |

### What Was Annotated

- ✅ **Green (immature) oranges** — counted
- ✅ **Transitioning yellow-green oranges** — Not counted
- ❌ **Fully ripe (orange-red) oranges** — excluded from GT
- ❌ **Flower buds or early-stage bud formations** — excluded

An orange was counted if at least 50% of its visible perimeter was present within the frame boundary, regardless of partial occlusion by leaves, branches, or neighbouring fruit.

### Annotation Format

**Excel files** (`ground_truth/30sec/` and `ground_truth/40sec/`):
- 30 files, one per video
- Columns: `image_filename`, `tree_id`, `video_id`, `frame_id`, `ground_truth_count`

**JSON progress files** (`ground_truth/progress/30sec/` and `ground_truth/progress/40sec/`):
- 30 files, one per video
- Fields: `treeId`, `videoId`, `totalFrames`, `frameNames`, `annotations` (dict of frame index → list of `{x, y, manual}` dot objects), `currentFrame`, `savedAt`
- 123,254 total hand-placed dot annotations with pixel coordinates

### Annotation Tool

Ground truth was produced using the **Orange Annotator** — a custom browser-based annotation tool included in this repository (`annotator/Orange_Annotator.html`). The annotator loads a folder of JPEG frames, allows placing dot markers by clicking on each orange, and auto-saves progress to JSON. Counts are exported to Excel upon completion.


---

## Models

### Detection-Based Models (Tiled Inference)

| Model | Backbone | Pretrained On | Venue | Input | Selected Param | Cal MAE |
|-------|----------|--------------|-------|-------|----------------|---------|
| **CutLER** | ResNet-50 + DINO ViT | ImageNet-1K | CVPR 2023 | 8 × 1920×1080 tiles | threshold = 0.7 | 8.69 |
| **CuVLER** | ResNet-50 + DINO ViT | ImageNet-1K | CVPR 2024 | 8 × 1920×1080 tiles | threshold = 0.5 | 7.94 |

Tiles merged with cross-tile NMS (IoU = 0.5). Overlap = 100px. Coverage = 100% (0 uncovered pixels).

### Feature Clustering Models (Full-Image Inference)

| Model | Backbone | Pretrained On | Venue | Input | Selected Param | Cal MAE |
|-------|----------|--------------|-------|-------|----------------|---------|
| **iBOT** | ViT-S/16 | ImageNet-1K | ICLR 2022 | 224px full-image | k = 9 | 6.52 |
| **MAE** | ViT-B/16 | ImageNet-1K | CVPR 2022 | 224px full-image | k = 9 | 5.70 |
| **DINOv2** | ViT-S/14 | LVD-142M | TMLR 2024 | 392px full-image | k = 6, min_area = 400 | 7.31 |
| **DETReg** | ResNet-50 (Def. DETR) | ImageNet+COCO | CVPR 2022 | 800/1333px full-image | threshold = 0.97 | 9.41 |

k-means clustering requires global scene context — tiling destroys cluster reference and produces MAE > 200.
DETReg issues 300 fixed queries per image — tiling produced 2,400 candidates per frame, MAE = 803.5 — rejected.

### Attention / Spectral Models

| Model | Backbone | Pretrained On | Venue | Input | Selected Param | Cal MAE |
|-------|----------|--------------|-------|-------|----------------|---------|
| **DINO** | ViT-S/8 | ImageNet-1K | ICCV 2021 | 480px full-image | thr = 0.3, min_area = 100 | 9.45 |
| **LOST** | ViT-S/8 | ImageNet-1K | BMVC 2021 | 128px sliding window | win=128, stride=42, nms=0.25 | 12.72 |

---

## The 4K Resolution Gap

All eight models were pretrained at approximately 224–800px. At default input resolution, a green orange spanning 50–100px in 4K shrinks to 3–21px — below every model's perceptual threshold.

| Setting | Scale | Orange Size | Result |
|---------|-------|-------------|--------|
| Default 800px resize | 0.208 | ~17px | **Model blind. Threshold has zero effect.** |
| Default 224px resize | 0.058 | ~6px | Below one patch token. Fails. |
| Tiled 1920×1080 | 0.500 | ~50px | Detectable. Functions correctly. |
| Full-image 392–480px | varies | 30–50px | Sufficient for clustering/attention. |
| Sliding window 128px | native | 60–100% | Visible. Seed-expansion works. |

> **Critical finding:** At 800px resize, a confidence threshold sweep across all values 0.1–0.9 returned **identical MAE at every threshold**, demonstrating the model detected nothing meaningful. This failure is silent — predictions are returned without error. Resolution adaptation is a prerequisite, not an optional enhancement.

---

## Inference Strategies

### Strategy 1 — Tiled Inference (CutLER, CuVLER)

Each 4K frame partitioned into 8 tiles (2 columns × 4 rows) of 1920×1080 pixels with 100px overlap. Detection per tile. Bounding boxes offset to full-frame coordinates. Cross-tile NMS (IoU=0.5).

```
Tiles per frame : 8
Overlap         : 100px
Coverage        : 100% — 0 uncovered pixels
Min coverage    : 1× per pixel
Max coverage    : 4× (overlap zones)
```

### Strategy 2 — Full-Image Resize (DETReg)

Full-image resize to 800/1333px. 300 fixed queries per image. Threshold = 0.97 selected on calibration frames. Tiled inference rejected: MAE = 803.5 at GT = 23.

### Strategy 3 — Full-Image Clustering (iBOT, MAE, DINOv2)

Full 4K frame resized to model's native input (224px for iBOT/MAE, 392px for DINOv2). k-means on patch features. Foreground connected components counted. Tiling rejected: MAE = 224 vs GT mean ~12.

### Strategy 4 — Full-Image Attention (DINO)

CLS token self-attention map at 480×480. Thresholded at 0.3. Connected components in binary mask counted.

### Strategy 5 — Sliding Window (LOST)

Window 128×128px, stride 42px. ~1 orange per window. Per-window detections mapped to full-frame. Cross-window NMS (IoU=0.25).

---

## Color Filter

Two-stage HSV filter applied post-inference to remove ripe orange detections (excluded from GT). No weights modified.

**Stage 1 — Ripe rejection:** Discard if ripe pixels (HSV: hue 15–40°, sat > 120, val > 160) exceed 8% of box area.

**Stage 2 — Green confirmation:** Discard if green pixels (HSV: hue 25–85°, sat > 40) fall below 25% of box area.

### Suppression Results

| Model | Suppressed | Failed | Rate |
|-------|------------|--------|------|
| LOST | 364 / 370 | 6 / 370 | 98.4%* |
| MAE | 336 / 370 | 34 / 370 | **90.8%** |
| DINOv2 | 314 / 370 | 56 / 370 | 84.9% |
| iBOT | 312 / 370 | 58 / 370 | 84.3% |
| CutLER | 290 / 370 | 80 / 370 | 78.4% |
| CuVLER | 237 / 370 | 133 / 370 | 64.1% |
| DETReg | 85 / 370 | 285 / 370 | 23.0% |
| DINO | 83 / 370 | 287 / 370 | 22.4% |

*LOST's 98.4% is an artifact — its boxes rarely overlap with any real fruit region (spatial F1 ≈ 0.000).

**Known structural limitation:** CutLER, CuVLER, and DETReg generate cluster-level boxes (mean area > 1.6M pixels). Ripe pixels diluted to < 5% of box area by surrounding foliage — below Stage 1 threshold. Structural consequence of cluster-level detection, not a filter design failure.

---

## Evaluation Metrics

### Count Accuracy (from Excel GT)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** (primary) | mean(\|pred − GT\|) | Average absolute error per frame. MAE=5.28 means ~5 oranges off per frame. |
| **RMSE** | sqrt(mean((pred−GT)²)) | Penalises large errors more than MAE |
| **W±2%** | % frames where \|pred−GT\| ≤ 2 | W±2%=30.2% means correct within 2 in ~1 of every 3 frames |
| **Bias** | mean(pred − GT) | Positive = overcounts, negative = undercounts. Bias=−0.64 ≈ zero systematic error |

### Spatial Accuracy (from JSON dot annotations, 123,254 points)

A predicted box is a **true positive (TP)** if at least one GT dot falls within its boundaries. Each dot and each box matched at most once.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | P=0.342 means 1 in 3 predicted boxes contains a real orange |
| **Recall** | TP / (TP + FN) | R=0.499 means model finds ~half of all real oranges |
| **F1** | 2×P×R / (P+R) | Harmonic mean of Precision and Recall |

### Reporting Levels

Results computed at five granularity levels:
1. Overall (10,577 frames)
2. Calibration vs Test split
3. Per-tree (10 trees)
4. Per-video (30 videos)
5. Group comparison (30-second vs 40-second trees)

---

## Setup

### Requirements

```bash
# Python 3.11, PyTorch 2.4.0, CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas openpyxl matplotlib opencv-python tqdm scipy scikit-learn tabulate timm
```

### Hardware

All experiments run on NVIDIA A100 80GB PCIe (85.1 GB VRAM), PyTorch 2.4.0, CUDA 12.1, Python 3.11.9.

---

## Running the Notebooks

### Step 1 — Shared Setup (run once)

```bash
jupyter notebook notebooks/00_shared.ipynb
```

Loads all 30 Excel + JSON files, defines calibration/test split, computes GT statistics, saves shared cache to `notebooks/cache/`.

### Step 2 — Run Each Model

```bash
jupyter notebook notebooks/01_CutLER_GreenOrange.ipynb
jupyter notebook notebooks/02_iBOT_GreenOrange.ipynb
jupyter notebook notebooks/03_MAE_GreenOrange.ipynb
jupyter notebook notebooks/04_DETReg_GreenOrange.ipynb
jupyter notebook notebooks/05_CuVLER_GreenOrange.ipynb
jupyter notebook notebooks/06_DINOv2_GreenOrange.ipynb
jupyter notebook notebooks/07_DINO_GreenOrange.ipynb
jupyter notebook notebooks/08_LOST_GreenOrange.ipynb
```

### Notebook Cell Structure (same across all models)

| Cell | Purpose |
|------|---------|
| Cell 1 | Configuration and paths |
| Cell 2 | Imports and dependencies |
| Cell 3 | Load model / download weights |
| Cell 4 | Load ground truth from shared cache |
| Cell 5 | Dataset inventory |
| Cell 6 | Color filter + inference functions |
| Cell 7 | Parameter sweep (calibration frames only) |
| Cell 8–9 | Full inference (10,577 frames) + spatial metrics pass |
| Cell 10 | Overall metrics (MAE, RMSE, W±2%, Bias) |
| Cell 11 | Per-video summary |
| Cell 12 | Per-tree summary |
| Cell 13 | Group comparison (30-sec vs 40-sec) |
| Cell 14 | Visualisations |
| Cell 15 | Ripe orange suppression validation |
| Cell 16 | Export all results to Excel (11 sheets) |

### Step 3 — Results

Each model saves an Excel file to `results/<model>/` with 11 sheets:

| Sheet | Contents |
|-------|----------|
| 1. Summary | Overall metrics + config |
| 2. Frame Results | 10,577 rows with pred, GT, error per frame |
| 3. Per-Video | 30 videos — MAE, RMSE, W±2%, Bias |
| 4. Per-Tree | 10 trees with CAL/TEST labels |
| 5. Group Comparison | 30-sec vs 40-sec trees |
| 6. Cal vs Test | Calibration vs test split metrics |
| 7. Param Sweep | Full sweep table with all candidate values |
| 8. Spatial Metrics | Precision/Recall/F1 per frame |
| 9. Spatial Per-Tree | Spatial metrics per tree |
| 10. Ripe Validation | Color filter suppression results |
| 11. Inventory | Frame counts per video |

---

## Key Findings

### 1. DINOv2 Achieves Best Count Accuracy
DINOv2 (MAE = 5.28, W±2% = 30.2%, Bias = −0.64) outperforms all other models on the test split. Its pretraining on the large-scale LVD-142M dataset produces richer patch features that cluster more reliably around orange-sized foreground regions than ImageNet-1K pretrained models.

### 2. The 4K Resolution Gap Is a Prerequisite Issue
All models return degenerate outputs at default input resolutions. Confidence threshold sweeps return identical MAE at all values 0.1–0.9 — the model is effectively blind. This failure is silent. Resolution adaptation is required before any meaningful evaluation.

### 3. Count Accuracy and Spatial Accuracy Are Dissociated
DETReg achieves the best spatial F1 (0.406) yet the second-worst MAE (8.11). DINOv2 achieves the best MAE (5.28) yet spatial F1 of only 0.142. Count accuracy can be achieved through coincidental cancellation of errors without accurate instance-level localisation.

### 4. Six of Eight Models Systematically Undercount
Systematic undercounting is the dominant failure mode. Models pretrained on ImageNet cannot distinguish small camouflaged oranges from spectrally identical foliage. Only DETReg (+4.58) and CuVLER (+1.92) overcount.

### 5. Zero-Shot Models Are Not Yet Deployment-Ready
The best model (DINOv2, MAE = 5.28) represents ~45% of the mean GT count. For practical yield estimation requiring 10–15% accuracy, no model achieves deployment-ready performance zero-shot.

### 6. Tree-Level Variability Dominates for High-Density Trees
Tree 07 (GT mean 19.74 — highest density) yields worst MAE across nearly all models. Dense clusters are systematically missed regardless of architecture.

---

## Ripe Validation Dataset

| Video | Tree | Frames | Ripe Points | Avg/Frame |
|-------|------|--------|-------------|-----------|
| Vid 02 | Tree 06 | 20 | 103 | 5.2 |
| Vid 03 | Tree 06 | 20 | 89 | 4.5 |
| Vid 02 | Tree 07 | 20 | 20 | 1.0 |
| Vid 03 | Tree 07 | 20 | 68 | 3.4 |
| Vid 02 | Tree 09 | 20 | 90 | 4.5 |
| **Total** | 3 trees | **100** | **370** | 3.7 |

---

## Citation

```bibtex
@article{greenorange2026,
  title   = {Self-Supervised Green Orange Counting via Object Discovery
             and Representation Learning},
  year    = {2026}
}
```

---

## License

The annotation files and evaluation code are released publicly to support future research in annotation-free agricultural vision. The full frame dataset will be released upon camera-ready submission.

---

## Acknowledgements

- [CutLER](https://github.com/facebookresearch/CutLER) — Wang et al., CVPR 2023
- [iBOT](https://github.com/bytedance/ibot) — Zhou et al., ICLR 2022
- [MAE](https://github.com/facebookresearch/mae) — He et al., CVPR 2022
- [DETReg](https://github.com/amirbar/DETReg) — Bar et al., CVPR 2022
- [CuVLER](https://github.com/shahaf-arica/CuVLER) — Arica et al., CVPR 2024
- [DINOv2](https://github.com/facebookresearch/dinov2) — Oquab et al., TMLR 2024
- [DINO](https://github.com/facebookresearch/dino) — Caron et al., ICCV 2021
- [LOST](https://github.com/valeoai/LOST) — Siméoni et al., BMVC 2021