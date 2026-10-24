# SupeRANSAC: One RANSAC to Rule them All ([Arxiv Paper](https://arxiv.org/abs/2506.04803))

## About

SupeRANSAC is a Python library that provides bindings for an advanced RANSAC C++ implementation using `pybind11`. 
It supports a wide variety of sampling, scoring, local optimization, degeneracy handling, and inlier selection techniques for robust model estimation tasks.
It provides estimators for **homography**, **essential matrix**, **fundamental matrix**, **6D rigid transformation**, and **6D absolute pose** estimation.

## Installation

Clone the repository and its submodules:
```bash
git clone git@github.com:danini/superansac.git
```

Make sure that you have the necessary OpenCV and Eigen libraries installed:
```bash
sudo apt-get update
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev cmake build-essential libboost-all-dev libeigen3-dev
```

Install SupeRANSAC by running:
```bash
pip install .
```

---

# Results

All benchmarks below evaluate **`SupeRANSAC`** using the default `RANSACSettings()` out of the box against **`PoseLib`** (for Fundamental matrix, Essential matrix, and Homography estimation) and **`OpenCV RANSAC`** (`cv2.solvePnPRansac` and `cv2.estimateAffine3D` for 6D Absolute Pose and 6D Rigid Registration) across **9 public multi-view and 3D benchmarks**: **PhotoTourism** (*CVPR 2020 RANSAC Tutorial*), **ETH3D** (*High-Res Multi-View*), **7Scenes** (*RGB-D indoor*), **Kusvod2**, **AdelaideRMF / Multi-H**, **Strecha Fountain**, **EVD** (*Extreme View Dataset*), **Homogr**, and **pose6dscene / 3D scan registration**.

Every table and figure in this README is directly reproducible via standalone test scripts in `tests/`:

```bash
# 0. Activate the Python virtual environment (if using ~/ransac/.venv on this machine)
source ~/ransac/.venv/bin/activate

# 1. Download & prepare all public benchmark datasets into data/
python tests/download_datasets.py

# 2. Experiment 1 & 2 (Relative Pose: Fundamental & Essential Matrix across 5 datasets)
python tests/test_relative_pose.py --problem all --features all

# 3. Experiment 1 & 2 (Homography across 5 datasets: EVD, Homogr, AdelaideRMF, ETH3D, PhotoTourism)
python tests/test_homography.py --features all

# 4. Experiment 3 (7 Feature Matchers: LoMa, RoMa, LoFTR, DISK+LG, RootSIFT, SIFT, ORB)
python tests/test_detectors.py --problem all

# 5. Experiment 4 (6D Absolute Pose PnP & 6D Rigid Registration SE(3) + Overview Figure)
python tests/test_6d_pose.py --problem all
```

---

## 1. Multi-Dataset Benchmark Overview

Across all 5 geometric estimation tasks and 9 datasets (`825` evaluation problems per matcher), `SupeRANSAC` consistently and substantially outperforms `PoseLib` and `OpenCV RANSAC`:

<p align="center">
  <img src="assets/benchmark_overview.png" width="820"/>
</p>

| Geometric Task | Datasets (`# Problems`) | Primary Metric | Baseline (`PoseLib` / `OpenCV`) | **`SupeRANSAC` (3-Matcher Avg)** | **`SupeRANSAC` (Best Matcher)** | Gain (Avg / Best) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fundamental Matrix ($F$)** | PhotoTourism, ETH3D, 7Scenes, Kusvod2/AdelaideRMF (`212` pairs) | Pose $\text{AUC}@10^\circ$ | `43.54%` (`PoseLib`) | **`54.34%`** | **`61.68%`** (`LoMa`, vs `47.81%`) | **`+10.80 pt` / `+13.87 pt`** |
| **Essential Matrix ($E$)** | PhotoTourism, ETH3D, 7Scenes, Strecha, Kusvod2 (`158` pairs) | Pose $\text{AUC}@10^\circ$ | `54.63%` (`PoseLib`) | **`60.62%`** | **`67.90%`** (`LoMa`, vs `61.22%`) | **`+6.00 pt` / `+6.68 pt`** |
| **Homography ($H$)** | EVD, Homogr, AdelaideRMF/Multi-H, ETH3D, PhotoTourism (`84` pairs) | Corner $\text{AUC}@5\text{px}$ | `25.65%` (`PoseLib`) | **`36.29%`** | **`44.25%`** (`RoMa`, vs `31.23%`) | **`+10.64 pt` / `+13.02 pt`** |
| **6D Absolute Pose ($PnP$)** | 7Scenes RGB-D 2D–3D, pose6dscene (`203` problems) | Pose $\text{AUC}@(10^\circ, 50\text{cm})$ | `45.49%` (`OpenCV RANSAC`) | **`78.56%`** | **`90.78%`** (`7Scenes RGB-D`) | **`+33.07 pt`** (**`5.1x` faster**) |
| **6D Rigid Pose ($SE(3)$)** | 7Scenes RGB-D 3D–3D, 3D Scan Benchmark (`168` problems) | Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `16.46%` (`OpenCV RANSAC`) | **`80.60%`** | **`84.99%`** (`3D Scan Benchmark`) | **`+64.14 pt`** |

### Per-Dataset Breakdown (`LoMa` Features)

#### Fundamental Matrix ($F$, `212` pairs across 4 datasets)
| Dataset | Pairs | Method | $\text{AUC}@5^\circ \uparrow$ | $\text{AUC}@10^\circ \uparrow$ | $\text{AUC}@20^\circ \uparrow$ | Med Error ($^\circ$) $\downarrow$ | Mean Time (ms) $\downarrow$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **PhotoTourism** | 38 | **`SupeRANSAC`** / `PoseLib` | **`60.48%`** / `42.90%` | **`74.45%`** / `56.79%` (**`+17.66 pt`**) | **`84.38%`** / `70.14%` | **`1.49°`** / `2.80°` | `51.49` / `28.83` |
| **ETH3D** | 39 | **`SupeRANSAC`** / `PoseLib` | **`67.99%`** / `56.92%` | **`81.14%`** / `70.25%` (**`+10.89 pt`**) | **`88.01%`** / `80.38%` | **`0.85°`** / `1.32°` | `62.06` / `38.92` |
| **7Scenes** | 45 | **`SupeRANSAC`** / `PoseLib` | **`15.30%`** / `13.49%` | **`37.10%`** / `33.83%` (**`+3.27 pt`**) | **`56.99%`** / `50.98%` | **`5.19°`** / `6.25°` | `72.61` / `28.07` |
| **Kusvod2 / AdelaideRMF** | 90 | **`SupeRANSAC`** / `PoseLib` | **`42.76%`** / `30.09%` | **`61.46%`** / `42.83%` (**`+18.63 pt`**) | **`74.16%`** / `58.18%` | **`2.81°`** / `5.23°` | `56.19` / `36.40` |
| **Overall ($F$)** | **212** | **`SupeRANSAC`** / `PoseLib` | **`44.10%`** / `33.14%` | **`61.68%`** / `47.81%` (**`+13.87 pt`**) | **`74.46%`** / `62.26%` | **`2.80°`** / `4.46°` | `59.91` / `33.74` |

#### Essential Matrix ($E$, `158` pairs across 5 datasets)
| Dataset | Pairs | Method | $\text{AUC}@5^\circ \uparrow$ | $\text{AUC}@10^\circ \uparrow$ | $\text{AUC}@20^\circ \uparrow$ | Med Error ($^\circ$) $\downarrow$ | Mean Time (ms) $\downarrow$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **PhotoTourism** | 30 | **`SupeRANSAC`** / `PoseLib` | **`81.51%`** / `80.08%` | **`87.42%`** / `86.88%` (**`+0.54 pt`**) | **`90.38%`** / `90.10%` | **`0.36°`** / `0.50°` | `75.38` / `42.99` |
| **ETH3D** | 30 | **`SupeRANSAC`** / `PoseLib` | `95.26%` / `95.48%` | `97.63%` / `97.74%` (`-0.11 pt`) | `98.81%` / `98.87%` | `0.24°` / `0.21°` | `79.81` / `38.97` |
| **7Scenes** | 36 | **`SupeRANSAC`** / `PoseLib` | `29.08%` / `30.20%` | **`55.14%`** / `52.30%` (**`+2.84 pt`**) | **`73.29%`** / `71.35%` | **`4.03°`** / `4.71°` | `76.96` / `34.99` |
| **Strecha / SacreCoeur** | 2 | **`SupeRANSAC`** / `PoseLib` | `48.01%` / `48.13%` | `49.00%` / `49.07%` (`-0.06 pt`) | `49.50%` / `49.53%` | `12.82°` / `12.56°` | `97.82` / `35.35` |
| **Kusvod2 / AdelaideRMF** | 60 | **`SupeRANSAC`** / `PoseLib` | **`33.02%`** / `20.21%` | **`52.44%`** / `37.14%` (**`+15.30 pt`**) | **`68.18%`** / `57.17%` | **`3.75°`** / `6.12°` | `88.89` / `56.01` |
| **Overall ($E$)** | **158** | **`SupeRANSAC`** / `PoseLib` | **`52.83%`** / `47.99%` | **`67.90%`** / `61.22%` (**`+6.68 pt`**) | **`78.89%`** / `74.14%` | **`2.10°`** / `2.29°` | `82.00` / `45.25` |

#### Homography ($H$, `84` pairs across 5 datasets)
| Dataset | Pairs | Method | $\text{AUC}@3\text{px} \uparrow$ | $\text{AUC}@5\text{px} \uparrow$ | $\text{AUC}@10\text{px} \uparrow$ | Med Error (px) $\downarrow$ | Mean Time (ms) $\downarrow$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **EVD** (*Extreme View*) | 22 | **`SupeRANSAC`** / `PoseLib` | **`46.86%`** / `31.68%` | **`65.65%`** / `40.84%` (**`+24.81 pt`**) | **`81.67%`** / `51.64%` | **`1.11 px`** / `4.05 px` | `30.45` / `4.39` |
| **Homogr** | 10 | **`SupeRANSAC`** / `PoseLib` | **`58.62%`** / `44.27%` | **`79.31%`** / `57.13%` (**`+22.18 pt`**) | **`89.66%`** / `63.57%` | **`0.99 px`** / `1.37 px` | `22.90` / `8.63` |
| **AdelaideRMF / Multi-H** | 22 | **`SupeRANSAC`** / `PoseLib` | **`6.56%`** / `0.00%` | **`7.83%`** / `0.00%` (**`+7.83 pt`**) | **`8.46%`** / `3.06%` | **`34.71 px`** / `39.05 px` | `24.78` / `2.17` |
| **ETH3D** | 15 | **`SupeRANSAC`** / `PoseLib` | **`25.56%`** / `24.00%` | **`40.97%`** / `40.51%` (**`+0.46 pt`**) | **`56.40%`** / `56.24%` | **`3.03 px`** / `3.04 px` | `14.14` / `10.02` |
| **PhotoTourism** | 15 | **`SupeRANSAC`** / `PoseLib` | **`10.79%`** / `10.74%` | **`12.06%`** / `12.04%` (**`+0.03 pt`**) | `12.70%` / `16.81%` | **`73.71 px`** / `75.67 px` | `14.24` / `6.33` |
| **Overall ($H$, `LoMa`)** | **84** | **`SupeRANSAC`** / `PoseLib` | **`25.77%`** / `18.74%` | **`37.15%`** / `26.33%` (**`+10.82 pt`**) | **`45.88%`** / `33.70%` | **`4.64 px`** / `25.81 px` | `22.26` / `5.66` |
| **Overall ($H$, `RoMa`)** | **84** | **`SupeRANSAC`** / `PoseLib` | **`29.32%`** / `21.64%` | **`44.25%`** / `31.23%` (**`+13.02 pt`**) | **`54.34%`** / `39.44%` | **`2.45 px`** / `9.98 px` | `27.45` / `18.87` |

### Exact Reproduction Commands (Section 1)
```bash
source ~/ransac/.venv/bin/activate
python tests/download_datasets.py
python tests/test_relative_pose.py --problem all --features all
python tests/test_homography.py --features all
python tests/test_6d_pose.py --problem all
```

---

## 2. Speed–Accuracy Trade-Off Across Datasets: `SupeRANSAC` vs. `PoseLib`

Below, we sweep the RANSAC iteration budget $b \in \{10, 25, 50, 100, 250, 500, 1000, 2500, 5000\}$ across the full multi-dataset benchmark (`212` pairs for $F$, `158` pairs for $E$, `84` pairs for $H$) for **DISK + LightGlue** (`splg`, top row), **RoMa** (`roma`, middle row), and **LoMa** (`loma`, bottom row) across **Essential Matrix** (left), **Fundamental Matrix** (middle), and **Homography** (right) estimation:

<table align="center">
  <tr>
    <td align="center"><img src="assets/essential_matrix_splg.png" width="260"/></td>
    <td align="center"><img src="assets/fundamental_matrix_splg.png" width="260"/></td>
    <td align="center"><img src="assets/homography_matrix_splg.png" width="260"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/essential_matrix_roma.png" width="260"/></td>
    <td align="center"><img src="assets/fundamental_matrix_roma.png" width="260"/></td>
    <td align="center"><img src="assets/homography_matrix_roma.png" width="260"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/essential_matrix_loma.png" width="260"/></td>
    <td align="center"><img src="assets/fundamental_matrix_loma.png" width="260"/></td>
    <td align="center"><img src="assets/homography_matrix_loma.png" width="260"/></td>
  </tr>
  <tr>
    <td align="center"><b>Essential Matrix (5 Datasets)</b></td>
    <td align="center"><b>Fundamental Matrix (4 Datasets)</b></td>
    <td align="center"><b>Homography (5 Datasets)</b></td>
  </tr>
</table>

### Exact Reproduction Commands (Section 2)
`test_relative_pose.py` and `test_homography.py` automatically evaluate the 9 iteration budgets and generate all 9 speed–accuracy plots in `assets/`:
```bash
source ~/ransac/.venv/bin/activate
python tests/test_relative_pose.py --problem all --features all
python tests/test_homography.py --features all
```

---

## 3. Multi-Detector Evaluation Across 7 Feature Matchers (including `LoMa`)

We evaluate **`SupeRANSAC`** and **`PoseLib`** across all multi-dataset pairs (`212` pairs for $F$, `158` pairs for $E$, `84` pairs for $H$) using **7 modern deep and classical feature matchers**: **[LoMa](https://github.com/davnords/LoMa)** (*ECCV 2026 Oral: Local Feature Matching Revisited*), **[RoMa](https://github.com/Parskatt/RoMa)**, **LoFTR**, **DISK + LightGlue**, **RootSIFT**, **SIFT**, and **ORB**. Across **every single detector and all three geometric tasks ($F$, $E$, and $H$)**, `SupeRANSAC` beats `PoseLib`:

<table align="center">
  <tr>
    <td align="center"><img src="assets/detector_fundamental.png" width="440"/></td>
    <td align="center"><img src="assets/detector_essential.png" width="440"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="assets/detector_homography.png" width="480"/></td>
  </tr>
</table>

### Fundamental Matrix ($F$, `212` pairs across PhotoTourism, ETH3D, 7Scenes, Kusvod2/AdelaideRMF)
`SupeRANSAC` outperforms `PoseLib` by **`+8.45 pt` to `+13.86 pt`** in $\text{AUC}@10^\circ$ (`+11.16 pt` average across all 7 matchers):

| Detector / Matcher | `PoseLib` ($\text{AUC}@5^\circ / 10^\circ / 20^\circ$) | **`SupeRANSAC`** ($\text{AUC}@5^\circ / 10^\circ / 20^\circ$) | `PoseLib` / **`SupeRANSAC`** Med Err | Gain in $\text{AUC}@10^\circ$ |
| :--- | :---: | :---: | :---: | :---: |
| **LoMa** | `33.14%` / `47.81%` / `62.26%` | **`44.10%` / `61.68%` / `74.46%`** | `4.46°` / **`2.80°`** | **`+13.86 pt`** |
| **RoMa** | `35.37%` / `50.03%` / `62.90%` | **`42.30%` / `58.48%` / `72.09%`** | `4.31°` / **`3.06°`** | **`+8.45 pt`** |
| **LoFTR** | `27.39%` / `40.31%` / `51.96%` | **`35.70%` / `49.96%` / `61.78%`** | `6.46°` / **`4.00°`** | **`+9.64 pt`** |
| **DISK + LightGlue** | `19.80%` / `32.77%` / `44.71%` | **`27.10%` / `42.87%` / `55.87%`** | `9.82°` / **`5.18°`** | **`+10.10 pt`** |
| **RootSIFT** | `23.31%` / `32.39%` / `41.80%` | **`33.59%` / `44.44%` / `54.35%`** | `13.73°` / **`4.97°`** | **`+12.05 pt`** |
| **SIFT** | `20.24%` / `29.07%` / `39.19%` | **`30.77%` / `42.15%` / `51.73%`** | `14.85°` / **`5.99°`** | **`+13.08 pt`** |
| **ORB** | `12.17%` / `20.03%` / `29.75%` | **`19.77%` / `30.99%` / `42.94%`** | `24.26°` / **`11.00°`** | **`+10.96 pt`** |
| **7-Matcher Average** | `24.49%` / `36.06%` / `47.51%` | **`33.33%` / `47.22%` / `59.03%`** | `11.13°` / **`5.29°`** | **`+11.16 pt`** |

### Essential Matrix ($E$, `158` pairs across PhotoTourism, ETH3D, 7Scenes, Strecha, Kusvod2/AdelaideRMF)
`SupeRANSAC` outperforms `PoseLib` across **all 7 matchers** (`+4.86 pt` average in $\text{AUC}@10^\circ$, up to **`+7.91 pt`**):

| Detector / Matcher | `PoseLib` ($\text{AUC}@5^\circ / 10^\circ / 20^\circ$) | **`SupeRANSAC`** ($\text{AUC}@5^\circ / 10^\circ / 20^\circ$) | `PoseLib` / **`SupeRANSAC`** Med Err | Gain in $\text{AUC}@10^\circ$ |
| :--- | :---: | :---: | :---: | :---: |
| **LoMa** | `47.99%` / `61.22%` / `74.14%` | **`52.83%` / `67.90%` / `78.89%`** | `2.29°` / **`2.10°`** | **`+6.68 pt`** |
| **RoMa** | `42.23%` / `57.20%` / `71.14%` | **`48.09%` / `65.11%` / `77.91%`** | `3.40°` / **`2.30°`** | **`+7.91 pt`** |
| **LoFTR** | `38.55%` / `50.38%` / `63.04%` | **`40.35%` / `55.29%` / `67.89%`** | `3.84°` / **`3.44°`** | **`+4.91 pt`** |
| **DISK + LightGlue** | `31.88%` / `45.45%` / `58.62%` | **`32.88%` / `48.58%` / `62.16%`** | `4.96°` / **`4.10°`** | **`+3.13 pt`** |
| **RootSIFT** | `33.32%` / `43.58%` / `55.15%` | **`35.40%` / `48.42%` / `61.16%`** | `5.91°` / **`4.74°`** | **`+4.84 pt`** |
| **SIFT** | `30.74%` / `41.29%` / `54.63%` | **`33.77%` / `45.58%` / `57.41%`** | `6.33°` / **`5.74°`** | **`+4.29 pt`** |
| **ORB** | `23.00%` / `36.30%` / `51.33%` | **`25.16%` / `38.58%` / `53.81%`** | `7.41°` / **`6.22°`** | **`+2.29 pt`** |
| **7-Matcher Average** | `35.39%` / `47.92%` / `61.15%` | **`38.36%` / `52.78%` / `65.61%`** | `4.88°` / **`4.09°`** | **`+4.86 pt`** |

### Homography Matrix ($H$, `84` pairs across EVD, Homogr, AdelaideRMF/Multi-H, ETH3D, PhotoTourism)
`SupeRANSAC` outperforms `PoseLib` across **all 7 matchers** (`+9.67 pt` average in $\text{AUC}@5\text{px}$, up to **`+13.02 pt`**):

| Detector / Matcher | `PoseLib` ($\text{AUC}@3 / 5 / 10\text{px}$) | **`SupeRANSAC`** ($\text{AUC}@3 / 5 / 10\text{px}$) | `PoseLib` / **`SupeRANSAC`** Med Err | Gain in $\text{AUC}@5\text{px}$ |
| :--- | :---: | :---: | :---: | :---: |
| **LoMa** | `18.74%` / `26.33%` / `33.70%` | **`25.77%` / `37.15%` / `45.88%`** | `25.81 px` / **`4.64 px`** | **`+10.82 pt`** |
| **RoMa** | `21.64%` / `31.23%` / `39.44%` | **`29.32%` / `44.25%` / `54.34%`** | `9.98 px` / **`2.45 px`** | **`+13.02 pt`** |
| **LoFTR** | `16.78%` / `23.85%` / `29.32%` | **`20.96%` / `32.12%` / `39.90%`** | `33.51 px` / **`10.75 px`** | **`+8.27 pt`** |
| **DISK + LightGlue** | `12.07%` / `19.38%` / `26.81%` | **`16.92%` / `27.47%` / `36.33%`** | `23.87 px` / **`13.53 px`** | **`+8.09 pt`** |
| **RootSIFT** | `14.38%` / `18.74%` / `22.33%` | **`20.23%` / `29.47%` / `37.22%`** | `81.65 px` / **`22.19 px`** | **`+10.73 pt`** |
| **SIFT** | `13.08%` / `17.22%` / `20.66%` | **`20.85%` / `29.54%` / `37.07%`** | `109.25 px` / **`20.69 px`** | **`+12.32 pt`** |
| **ORB** | `8.85%` / `14.19%` / `18.96%` | **`10.49%` / `18.65%` / `27.75%`** | `81.28 px` / **`23.81 px`** | **`+4.46 pt`** |
| **7-Matcher Average** | `15.08%` / `21.56%` / `27.32%` | **`20.65%` / `31.23%` / `39.78%`** | `52.19 px` / **`13.99 px`** | **`+9.67 pt`** |

### Exact Reproduction Command (Section 3)
```bash
source ~/ransac/.venv/bin/activate
python tests/test_detectors.py --problem all
```

---

## 4. 6D Absolute Pose ($PnP$) & 6D Rigid Registration ($SE(3)$): `SupeRANSAC` vs. `OpenCV RANSAC`

We compare **`SupeRANSAC`** against **`OpenCV RANSAC`** (`cv2.solvePnPRansac` with matched `5000` iteration budget for 2D–3D absolute pose, and `cv2.estimateAffine3D` + SVD orthogonalization for 3D–3D rigid registration) across **both real RGB-D indoor scenes (`7Scenes RGB-D`) and the 6D pose / point cloud registration benchmarks (`pose6dscene`, `rigid_pose_example`)**:

### 6D Absolute Pose Estimation (2D–3D $PnP$, `203` problems)

Across `7Scenes RGB-D` (`63` real RGB-D 2D–3D problems with `LoMa`, `RoMa`, and `DISK+LG` matches lifted via sensor depth maps) and `pose6dscene` (`140` problems across 7 outlier ratios from `33%` to `91%`), `SupeRANSAC` achieves **`78.56%`** Pose $\text{AUC}@(10^\circ, 50\text{cm})$ at **`46.77 ms`** mean runtime compared to `OpenCV RANSAC`'s **`45.49%`** at **`236.42 ms`** (**`+33.07 pt` higher accuracy** and **`5.1x` faster**):

<p align="center">
  <img src="assets/absolute_pose_experiments.png" width="840"/>
</p>

| Dataset | Problems | Method | $\text{AUC}@(5^\circ,25\text{cm}) \uparrow$ | $\text{AUC}@(10^\circ,50\text{cm}) \uparrow$ | $\text{AUC}@(20^\circ,1\text{m}) \uparrow$ | Med Rot ($^\circ$) $\downarrow$ | Med Trans (cm) $\downarrow$ | Mean Time (ms) $\downarrow$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **7Scenes RGB-D** | 63 | **`SupeRANSAC`** | **`81.55%`** | **`90.78%`** | **`95.39%`** | **`0.65°`** | **`1.60 cm`** | **`33.37`** |
| **7Scenes RGB-D** | 63 | `OpenCV RANSAC` | `42.13%` | `51.85%` | `59.56%` | `3.29°` | `6.98 cm` | `288.81` |
| **pose6dscene** (`33%–91%` outliers) | 140 | **`SupeRANSAC`** | **`48.75%`** | **`73.19%`** | **`86.60%`** | **`1.51°`** | **`8.28 cm`** | **`52.79`** |
| **pose6dscene** (`33%–91%` outliers) | 140 | `OpenCV RANSAC` | `25.00%` | `42.89%` | `60.44%` | `2.52°` | `18.94 cm` | `212.84` |
| **Overall ($PnP$)** | **203** | **`SupeRANSAC`** | **`58.75%`** | **`78.56%`** (**`+33.07 pt`**) | **`89.28%`** | **`1.28°`** | **`5.54 cm`** | **`46.77`** (**`5.1x` faster**) |
| **Overall ($PnP$)** | **203** | `OpenCV RANSAC` | `30.12%` | `45.49%` | `60.00%` | `2.67°` | `16.20 cm` | `236.42` |

### 6D Rigid Transformation Estimation (3D–3D $SE(3)$, `168` problems)

Across `7Scenes RGB-D` (`68` real 3D–3D RGB-D registration problems) and `rigid_pose_example` (`100` 3D point cloud registration problems across 4 difficulty tiers), `SupeRANSAC` achieves **`80.60%`** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ (`0.57°` median rotation error, `1.54 cm` median translation error) compared to `OpenCV RANSAC`'s **`16.46%`** (`15.18°` median rotation error, `30.02 cm` median translation error, **`+64.14 pt`**):

<p align="center">
  <img src="assets/rigid_transform_experiments.png" width="840"/>
</p>

| Dataset | Problems | Method | $\text{AUC}@(2.5^\circ,5\text{cm}) \uparrow$ | $\text{AUC}@(5^\circ,10\text{cm}) \uparrow$ | $\text{AUC}@(10^\circ,20\text{cm}) \uparrow$ | Med Rot ($^\circ$) $\downarrow$ | Med Trans (cm) $\downarrow$ | Mean Time (ms) $\downarrow$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **7Scenes RGB-D** | 68 | **`SupeRANSAC`** | **`54.68%`** | **`74.51%`** | **`87.22%`** | **`0.79°`** | **`1.67 cm`** | `13.20` |
| **7Scenes RGB-D** | 68 | `OpenCV RANSAC` | `12.21%` | `18.38%` | `23.82%` | `20.99°` | `41.41 cm` | `10.28` |
| **rigid_pose_example** (4 Tiers) | 100 | **`SupeRANSAC`** | **`69.83%`** | **`84.99%`** | **`92.50%`** | **`0.51°`** | **`1.40 cm`** | `11.72` |
| **rigid_pose_example** (4 Tiers) | 100 | `OpenCV RANSAC` | `9.67%` | `15.62%` | `26.92%` | `12.65°` | `20.06 cm` | `10.38` |
| **Overall ($SE(3)$)** | **168** | **`SupeRANSAC`** | **`63.50%`** | **`80.60%`** (**`+64.14 pt`**) | **`90.28%`** | **`0.57°`** | **`1.54 cm`** | `12.32` |
| **Overall ($SE(3)$)** | **168** | `OpenCV RANSAC` | `10.44%` | `16.46%` | `25.38%` | `15.18°` | `30.02 cm` | `10.34` |

### Exact Reproduction Command (Section 4)
```bash
source ~/ransac/.venv/bin/activate
python tests/test_6d_pose.py --problem all
```

---

# Jupyter Notebook Examples

- **Homography Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_homography_fitting_splg.ipynb)

- **Homography Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_homography_fitting_roma.ipynb)

- **Fundamental Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_splg.ipynb)

- **Fundamental Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_roma.ipynb)

- **Essential Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_splg.ipynb)

- **Essential Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_roma.ipynb)

- **Absolute Pose Estimation**  
  Example available at: [notebook](examples/example_absolute_pose_fitting.ipynb)

- **6D Rigid Transformation Estimation**  
  Example available at: [notebook](examples/example_rigid_transform_fitting.ipynb)

---

## Usage

The library provides several model estimation functions and settings that allow customization of the RANSAC pipeline. `RANSACSettings()` is pre-configured with the best defaults for all estimators.

### Importing the Library

```python
import numpy as np
import pysuperansac
from pysuperansac import (
    ScoringType,
    SamplerType,
    LocalOptimizationType,
    InlierSelectorType,
    NeighborhoodType,
    CameraType,
    RANSACSettings,
)
```

### Example: Estimate a Homography, Fundamental, or Essential Matrix

```python
# Correspondences as an (N, 4) float64 array: [x1, y1, x2, y2], sorted by descending confidence
correspondences = np.array([
    [x1, y1, x2, y2],
    [x3, y3, x4, y4],
    # ...
], dtype=np.float64)
probabilities = np.ones(len(correspondences), dtype=np.float64)
image_sizes = [height1, width1, height2, width2]

# Default RANSACSettings() uses the best configuration automatically
config = RANSACSettings()

# Estimate Homography
H, inliers, score, iterations = pysuperansac.estimateHomography(
    correspondences, image_sizes, probabilities, config=config
)

# Estimate Fundamental Matrix
F, inliers, score, iterations = pysuperansac.estimateFundamentalMatrix(
    correspondences, image_sizes, probabilities, config=config
)

# Estimate Essential Matrix
E, inliers, score, iterations = pysuperansac.estimateEssentialMatrix(
    correspondences, K1, K2, image_sizes, probabilities, config=config
)
```

### Supported Estimation Functions

#### 1. Homography Estimation
- **Function**: `pysuperansac.estimateHomography(correspondences, image_sizes, probabilities=[], config=RANSACSettings())`
- **Description**: Estimates a $3 \times 3$ homography matrix from 2D-2D point correspondences.
- **Parameters**:
  - `correspondences`: `(N, 4)` `np.ndarray` (`[x1, y1, x2, y2]`).
  - `image_sizes`: `[height_src, width_src, height_dst, width_dst]`.
  - `probabilities` *(optional)*: Length-`N` list/array of prior match confidences.
  - `config`: An instance of `RANSACSettings`.

#### 2. Fundamental Matrix Estimation
- **Function**: `pysuperansac.estimateFundamentalMatrix(correspondences, image_sizes, probabilities=[], config=RANSACSettings())`
- **Description**: Estimates a $3 \times 3$ fundamental matrix from 2D-2D point correspondences with DEGENSAC and Plane-and-Parallax support.
- **Parameters**:
  - `correspondences`: `(N, 4)` `np.ndarray` (`[x1, y1, x2, y2]`).
  - `image_sizes`: `[height_src, width_src, height_dst, width_dst]`.
  - `probabilities` *(optional)*: Length-`N` list/array of prior match confidences.
  - `config`: An instance of `RANSACSettings`.

#### 3. Essential Matrix Estimation
- **Function**: `pysuperansac.estimateEssentialMatrix(correspondences, intrinsics_src, intrinsics_dst, image_sizes, probabilities=[], config=RANSACSettings())`
- **Description**: Estimates a $3 \times 3$ essential matrix using 2D-2D point correspondences and camera intrinsic matrices.
- **Parameters**:
  - `correspondences`: `(N, 4)` `np.ndarray` (`[x1, y1, x2, y2]`).
  - `intrinsics_src`: $3 \times 3$ source camera intrinsic matrix.
  - `intrinsics_dst`: $3 \times 3$ destination camera intrinsic matrix.
  - `image_sizes`: `[height_src, width_src, height_dst, width_dst]`.
  - `probabilities` *(optional)*: Length-`N` list/array of prior match confidences.
  - `config`: An instance of `RANSACSettings`.

#### 4. Rigid Transformation Estimation
- **Function**: `pysuperansac.estimateRigidTransform(correspondences, bounding_box_sizes, probabilities=[], config=RANSACSettings())`
- **Description**: Estimates a $4 \times 4$ 6D rigid transformation matrix from 3D-3D point correspondences.
- **Parameters**:
  - `correspondences`: `(N, 6)` `np.ndarray` (`[X1, Y1, Z1, X2, Y2, Z2]`).
  - `bounding_box_sizes`: `[max_x1, max_y1, max_z1, max_x2, max_y2, max_z2]`.
  - `probabilities` *(optional)*: Length-`N` list/array of prior match confidences.
  - `config`: An instance of `RANSACSettings`.

#### 5. Absolute Pose Estimation
- **Function**: `pysuperansac.estimateAbsolutePose(correspondences, camera_type, camera_params, bounding_box, probabilities=[], config=RANSACSettings())`
- **Description**: Estimates the 6D absolute pose $(R, t)$ of a camera from 2D-3D correspondences.
- **Parameters**:
  - `correspondences`: `(N, 5)` `np.ndarray` (`[u, v, X, Y, Z]`).
  - `camera_type`: Type of the camera (e.g., `CameraType.SimpleRadial`, `CameraType.SimplePinhole`).
  - `camera_params`: Camera intrinsic/distortion parameters.
  - `bounding_box`: `[height, width, max_X, max_Y, max_Z]`.
  - `probabilities` *(optional)*: Length-`N` list/array of prior match confidences.
  - `config`: An instance of `RANSACSettings`.

### Advanced Configuration: `RANSACSettings`

The `RANSACSettings` class provides fine-grained control over the RANSAC pipeline (shown below with the default values):

```python
config = RANSACSettings()
config.min_iterations = 1000
config.max_iterations = 5000
config.inlier_threshold = 1.5
config.confidence = 0.999999
config.scoring = ScoringType.MAGSAC
config.sampler = SamplerType.PROSAC
config.local_optimization = LocalOptimizationType.GCRANSAC
config.final_optimization = LocalOptimizationType.LSQ
config.neighborhood = NeighborhoodType.Grid
```

### Enumerations and Their Values

**Scoring Types**
- `ScoringType.RANSAC`
- `ScoringType.MSAC`
- `ScoringType.MAGSAC`
- `ScoringType.ACRANSAC`
- `ScoringType.GRID`

**Sampler Types**
- `SamplerType.Uniform`
- `SamplerType.PROSAC`
- `SamplerType.NAPSAC`
- `SamplerType.ProgressiveNAPSAC`
- `SamplerType.ImportanceSampler`
- `SamplerType.ARSampler`

**Local Optimization Types**
- `LocalOptimizationType.Nothing`
- `LocalOptimizationType.LSQ`
- `LocalOptimizationType.IRLS`
- `LocalOptimizationType.NestedRANSAC`
- `LocalOptimizationType.GCRANSAC`
- `LocalOptimizationType.IteratedLM`

**Neighborhood Types**
- `NeighborhoodType.Grid`
- `NeighborhoodType.FLANN_KNN`
- `NeighborhoodType.FLANN_Radius`
- `NeighborhoodType.BruteForce`

**Camera Types**
- `CameraType.SimpleRadial`
- `CameraType.SimplePinhole`

---

## Evaluation - Python

All evaluation scripts are self-contained and located in `tests/`:

1. **Download & Prepare Public Benchmark Datasets**:
   ```bash
   python tests/download_datasets.py --dataset all
   ```
   Downloads and prepares **PhotoTourism** (*CVPR 2020 RANSAC Tutorial*), **ETH3D** (*High-Res Multi-View*), **7Scenes** (*RGB-D indoor*), **Kusvod2**, **AdelaideRMF / Multi-H**, **Strecha Fountain**, **EVD**, **Homogr**, **pose6dscene**, and **rigid_pose_example** into `data/`.

2. **Relative Pose Benchmark (`Fundamental` & `Essential` Matrix)**:
   ```bash
   # Evaluate both F and E across all datasets and matchers (LoMa, RoMa, DISK+LG)
   python tests/test_relative_pose.py --problem all --features all

   # Or filter by problem, dataset, or feature matcher:
   python tests/test_relative_pose.py --problem fundamental --dataset phototourism --features LoMa
   python tests/test_relative_pose.py --problem essential --dataset eth3d --features RoMa
   ```

3. **Homography Benchmark**:
   ```bash
   # Evaluate Homography across EVD, Homogr, AdelaideRMF/Multi-H, ETH3D, PhotoTourism
   python tests/test_homography.py --features all

   # Or evaluate a specific dataset / matcher:
   python tests/test_homography.py --dataset evd --features LoMa
   ```

4. **7-Detector Benchmark (`LoMa`, `RoMa`, `LoFTR`, `DISK+LG`, `RootSIFT`, `SIFT`, `ORB`)**:
   ```bash
   python tests/test_detectors.py --problem all
   ```

5. **6D Absolute Pose ($PnP$) & 6D Rigid Registration ($SE(3)$) Benchmark**:
   ```bash
   python tests/test_6d_pose.py --problem all
   ```

### Install Feature Detectors

To use **LoMa** ([ECCV 2026 Oral] *LoMa: Local Feature Matching Revisited*):
```bash
git clone https://github.com/davnords/LoMa.git && cd LoMa
pip install -e .
```

To use **SuperPoint / DISK + LightGlue** or **LoFTR** features, install `kornia` / `LightGlue`:
```bash
pip install kornia
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

To use **RoMa** features:
```bash
git clone https://github.com/Parskatt/RoMa && cd RoMa
pip install -e .
```

