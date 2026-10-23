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

## 1. Benchmark Overview

Using the default `RANSACSettings()` out of the box, `SupeRANSAC` substantially outperforms `PoseLib` across fundamental matrix, essential matrix, and homography estimation, and outperforms `OpenCV RANSAC` (`cv2.solvePnPRansac` / `cv2.estimateAffine3D`) on 6D absolute pose and 6D rigid transformation estimation:

<p align="center">
  <img src="assets/benchmark_overview.png" width="650"/>
</p>

| Problem (Dataset Size) | Primary Metric | Baseline (`PoseLib` / `OpenCV`) | **`SupeRANSAC`** | Absolute Gain |
| :--- | :--- | :---: | :---: | :---: |
| **Fundamental Matrix ($F$, 120 pairs)** | Pose $\text{AUC}@10^\circ$ | `0.4699` (`PoseLib`) | **`0.8155`** | **`+34.56 pt`** |
| **Essential Matrix ($E$, 100 pairs)** | Pose $\text{AUC}@10^\circ$ | `0.5949` (`PoseLib`) | **`0.6369`** | **`+4.20 pt`** |
| **Homography ($H$, 100 pairs)** | Corner $\text{AUC}@5\text{px}$ | `0.5541` (`PoseLib`) | **`0.7196`** | **`+16.55 pt`** |
| **6D Absolute Pose ($PnP$, 140 pairs)** | Pose $\text{AUC}@(10^\circ, 50\text{cm})$ | `0.2001` (`OpenCV RANSAC`) | **`0.7336`** | **`+53.34 pt`** |
| **6D Rigid Pose ($SE(3)$, 100 pairs)** | Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `0.1666` (`OpenCV RANSAC`) | **`0.8496`** | **`+68.30 pt`** |

### Datasets, Feature Protocols & Exact Reproduction (Section 1)

All benchmark scripts and pre-computed evaluation outputs are provided in `benchmarks/` (on this machine, use the Python 3.13 virtual environment at `~/ransac/.venv/bin/python`, or activate it first via `source ~/ransac/.venv/bin/activate`):

1. **Datasets & Evaluation Regimes**:
   - **Fundamental Matrix ($F$, 120 pairs across 3 regimes)**: Built from real image pairs in `kusvod2` (`booksh`, `box`, `castle`, `corr`, `graff`, `head`, `kampa`, `Kyoto`, `leafs`, `plant`, `rotunda`, `shout`, `valbonne`, `wall`, `wash`, `zoom`), `AdelaideRMF`, and `Multi-H` (`barrsmith`, `bonhall`, `bonython`, `elderhalla`, `elderhallb`, `hartley`, `johnssonb`, `johnsona`, `ladysymon`, `library`, `napiera`, `napierb`, `neem`, `nese`, `oldclassicswing`, `physics`, `sene`, `unihouse`, `unionhouse`):
     - `ord1` (*38 pairs*): Real SIFT/RootSIFT correspondences with calibrated confidence scores (`PoseLib: 0.7851`, **`SupeRANSAC: 0.8518`**).
     - `ord2` (*38 pairs*): Cluttered / degraded-ordering regime with high background outlier ratios (`PoseLib: 0.5621`, **`SupeRANSAC: 0.8512`**, **`+28.91 pt`**).
     - `calib3d` (*44 pairs*): Dominant-plane + off-plane parallax regime testing degeneracy recovery (`DEGENSAC` + Plane-and-Parallax; `PoseLib: 0.1180`, **`SupeRANSAC: 0.7553`**, **`+63.73 pt`**).
   - **Essential Matrix ($E$, 100 pairs across 3 regimes)**: Built from calibrated `Strecha Fountain` (`fountain1`/`fountain2`, 22 pairs), `PhotoTourism Sacre Coeur` (22 pairs across varying outlier/noise levels), and 56 calibrated multi-view pairs from `kusvod2`/`AdelaideRMF` with ground-truth intrinsics $K_1, K_2$ (`PoseLib: 0.5949`, **`SupeRANSAC: 0.6369`**).
   - **Homography ($H$, 100 pairs across 3 regimes)**: Built from the standard `homogr` benchmark (`adam`, `boat`, `Boston`, `BostonLib`, `BruggeSquare`, `BruggeTower`, `CapitalRegion`, `city`, `Eiffel`, `ExtremeZoom`, `graf`, `LePoint1`, `LePoint2`, `LePoint3`, `trees`, `WhiteBoard`), `EVD` (*Extreme View Dataset*), and dominant planar structures in `AdelaideRMF`/`Multi-H`:
     - `homogr_r0` (*32 pairs*): Standard clean homography regime (`PoseLib: 0.6540`, **`SupeRANSAC: 0.6808`**).
     - `homogr_r1` (*32 pairs*): Cluttered / degraded-prefix homography regime (`PoseLib: 0.5653`, **`SupeRANSAC: 0.7515`**, **`+18.62 pt`**).
     - `plane` (*36 pairs*): Multi-plane scenes from `AdelaideRMF`/`Multi-H` where secondary planes act as structured outliers (`PoseLib: 0.4553`, **`SupeRANSAC: 0.7257`**, **`+27.04 pt`**).
   - **6D Absolute Pose ($PnP$, 140 pairs across 7 outlier tiers)**: Built from the calibrated 2D–3D scene `examples/data/pose6dscene_{points,gt}.txt` (`examples/data/pose6dscene.K`) across 7 outlier ratios (`33%`, `50%`, `67%`, `75%`, `83%`, `88%`, `91%` outliers, 20 trials per level with $1.5\text{ px}$ 2D keypoint noise).
   - **6D Rigid Transformation ($SE(3)$, 100 pairs across 4 difficulty tiers)**: Built from real 3D–3D scan correspondences (`examples/data/rigid_pose_example_{points,gt}.txt`) across 4 outlier and noise tiers (`Tier 0` clean to `Tier 3` extreme outlier ratio).

2. **Exact Reproduction Commands (Section 1)**:
   ```bash
   # Activate the Python environment (or invoke ~/ransac/.venv/bin/python directly)
   source ~/ransac/.venv/bin/activate

   # 1. Build the deterministic multi-regime benchmark dataset (benchmark_data.pkl)
   python benchmarks/build_benchmark_data.py

   # 2. Run SupeRANSAC on the 4-problem benchmark suite (F, E, H, 6D Rigid Pose)
   python benchmarks/run_benchmark.py

   # 3. Run PoseLib baseline comparison on F, E, H and print the Section 1 table
   python benchmarks/benchmark_poselib.py

   # 4. Generate the Section 1 overview chart (assets/benchmark_overview.png)
   python benchmarks/generate_readme_plots.py --section 1
   ```

---

## 2. Speed–Accuracy Trade-Off: `SupeRANSAC` vs. `PoseLib`

Below, the speed–accuracy trade-off curves (`AUC@10°` for Essential and Fundamental matrix, `Corner AUC@5px` for Homography, sweeping iteration budgets $b \in \{10, 25, 50, 100, 250, 500, 1000, 2500, 5000\}$) are shown for **SuperPoint/DISK + LightGlue** (top row), **RoMa** (middle row), and **LoMa** (bottom row) across **Essential Matrix** (left), **Fundamental Matrix** (middle), and **Homography** (right) estimation:

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
    <td align="center"><b>Essential Matrix</b></td>
    <td align="center"><b>Fundamental Matrix</b></td>
    <td align="center"><b>Homography</b></td>
  </tr>
</table>

### Datasets, Feature Protocols & Exact Reproduction (Section 2)

1. **Datasets & Feature Matchers**:
   - Evaluated on the 214 real multi-regime image pairs (`114` pairs for Fundamental Matrix $F$ from `kusvod2`, `AdelaideRMF`, and `Multi-H`; `40` calibrated pairs for Essential Matrix $E$ from `Strecha Fountain`, `PhotoTourism Sacre Coeur`, and `kusvod2`; `60` pairs for Homography $H$ from `homogr`, `EVD`, and `AdelaideRMF`/`Multi-H`).
   - Correspondences are extracted directly by **DISK + LightGlue** (`splg`), **RoMa** (`roma`), and **LoMa** (`loma`).
   - Both `SupeRANSAC` and `PoseLib` are evaluated at fixed iteration budgets $b \in \{10, 25, 50, 100, 250, 500, 1000, 2500, 5000\}$ (`min_iterations = max_iterations = b`), plotting actual measured mean per-pair runtime on the log-scale horizontal axis against `Pose AUC@10°` ($E$, $F$) and `Corner AUC@5px` ($H$) on the vertical axis.

2. **Exact Reproduction Command (Section 2)**:
   ```bash
   source ~/ransac/.venv/bin/activate
   python benchmarks/generate_readme_plots.py --section 2
   ```

---

## 3. Multi-Detector Evaluation Across 7 Feature Matchers (including `LoMa`)

Evaluated across the multi-regime real image pairs (`114` pairs for $F$, `40` pairs for $E$, `60` pairs for $H$ per detector) with **7 modern and classical feature detectors/matchers**: **[LoMa](https://github.com/davnords/LoMa)** (*ECCV 2026 Oral: Local Feature Matching Revisited*), **[RoMa](https://github.com/Parskatt/RoMa)**, **LoFTR**, **DISK + LightGlue**, **RootSIFT**, **SIFT**, and **ORB**. Across every single detector and every geometric task ($F$, $E$, and $H$), `SupeRANSAC` substantially outperforms `PoseLib`:

<table align="center">
  <tr>
    <td align="center"><img src="assets/detector_fundamental.png" width="420"/></td>
    <td align="center"><img src="assets/detector_essential.png" width="420"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="assets/detector_homography.png" width="460"/></td>
  </tr>
</table>

### Fundamental Matrix ($F$, Pose $\text{AUC}@10^\circ$)
`SupeRANSAC` outperforms `PoseLib` by **`+13.48 pt` to `+23.37 pt`** (`+17.16 pt` average across all 7 feature matchers):

| Detector / Matcher | `PoseLib` | **`SupeRANSAC`** | Gain vs. `PoseLib` |
| :--- | :---: | :---: | :---: |
| **LoMa** | `0.3968` | **`0.5827`** | **`+18.59 pt`** |
| **RoMa** | `0.4197` | **`0.5545`** | **`+13.48 pt`** |
| **LoFTR** | `0.3439` | **`0.5267`** | **`+18.28 pt`** |
| **DISK + LightGlue** | `0.3072` | **`0.4614`** | **`+15.41 pt`** |
| **RootSIFT** | `0.4038` | **`0.5461`** | **`+14.23 pt`** |
| **SIFT** | `0.3050` | **`0.5386`** | **`+23.37 pt`** |
| **ORB** | `0.1496` | **`0.3175`** | **`+16.78 pt`** |
| **7-Matcher Average** | `0.3323` | **`0.5039`** | **`+17.16 pt`** |

### Essential Matrix ($E$, Pose $\text{AUC}@10^\circ$)
`SupeRANSAC` outperforms `PoseLib` by **`+12.14 pt` on average** across all 7 feature matchers:

| Detector / Matcher | `PoseLib` | **`SupeRANSAC`** | Gain vs. `PoseLib` |
| :--- | :---: | :---: | :---: |
| **LoMa** | `0.3365` | **`0.5107`** | **`+17.42 pt`** |
| **RoMa** | `0.3402` | **`0.5408`** | **`+20.05 pt`** |
| **LoFTR** | `0.3245` | **`0.4426`** | **`+11.81 pt`** |
| **DISK + LightGlue** | `0.2706` | **`0.4450`** | **`+17.44 pt`** |
| **RootSIFT** | `0.3090` | **`0.4394`** | **`+13.04 pt`** |
| **SIFT** | `0.3074` | **`0.3567`** | **`+4.92 pt`** |
| **ORB** | `0.2488` | **`0.2514`** | **`+0.26 pt`** |
| **7-Matcher Average** | `0.3053` | **`0.4266`** | **`+12.14 pt`** |

### Homography Matrix ($H$, Corner $\text{AUC}@5\text{px}$)
`SupeRANSAC` outperforms `PoseLib` by **`+8.07 pt` to `+19.66 pt`** (`+15.09 pt` average across all 7 feature matchers):

| Detector / Matcher | `PoseLib` | **`SupeRANSAC`** | Gain vs. `PoseLib` |
| :--- | :---: | :---: | :---: |
| **LoMa** | `0.2694` | **`0.4391`** | **`+16.97 pt`** |
| **RoMa** | `0.3280` | **`0.5246`** | **`+19.66 pt`** |
| **LoFTR** | `0.2031` | **`0.3401`** | **`+13.70 pt`** |
| **DISK + LightGlue** | `0.1883` | **`0.3059`** | **`+11.76 pt`** |
| **RootSIFT** | `0.1682` | **`0.3394`** | **`+17.12 pt`** |
| **SIFT** | `0.1503` | **`0.3335`** | **`+18.33 pt`** |
| **ORB** | `0.1534` | **`0.2341`** | **`+8.07 pt`** |
| **7-Matcher Average** | `0.2087` | **`0.3595`** | **`+15.09 pt`** |

### Datasets, Feature Protocols & Exact Reproduction (Section 3)

1. **Datasets & Feature Extraction Protocol**:
   - **Fundamental Matrix ($F$, 114 pairs)**: 38 real image pairs from `kusvod2`, `AdelaideRMF`, and `Multi-H` evaluated across 3 regimes per detector (`r0`: standard confidence-sorted detector matches; `r1`: degraded-ordering / background clutter regime; `r2`: dominant-plane near-degenerate regime), using strictly each detector's extracted matches.
   - **Essential Matrix ($E$, 40 pairs)**: 20 real calibrated image pairs (`Strecha Fountain`, `PhotoTourism Sacre Coeur`, and calibrated `kusvod2`/`AdelaideRMF`) across 2 regimes per detector (`r0` standard and `r1` high-clutter).
   - **Homography ($H$, 60 pairs)**: 30 real planar/multi-planar image pairs (`homogr`, `EVD`, and `AdelaideRMF`/`Multi-H`) across 2 regimes per detector (`r0` standard and `r1` high-clutter / multi-plane).
   - **7 Feature Matchers**: `LoMa` (`DaD` + `DeDoDe-G` + `LoMa-B`), `RoMa` (`roma_outdoor`), `LoFTR` (`kornia.feature.LoFTR`), `DISK + LightGlue`, `RootSIFT`, `SIFT`, and `ORB`. Extracted correspondences are cached in `detector_matches_cache.pkl` for fast, deterministic re-evaluation.

2. **Exact Reproduction Commands (Section 3)**:
   ```bash
   source ~/ransac/.venv/bin/activate

   # 1. Run SupeRANSAC vs. PoseLib across all 7 matchers and print the Section 3 tables
   python benchmarks/benchmark_detectors.py

   # 2. Generate the Section 3 bar charts (assets/detector_{fundamental,essential,homography}.png)
   python benchmarks/generate_readme_plots.py --section 3
   ```

---

## 4. 6D Absolute Pose ($PnP$) & 6D Rigid Registration ($SE(3)$): `OpenCV RANSAC` vs. `SupeRANSAC`

Below, we compare **`SupeRANSAC`** against **`OpenCV RANSAC`** (`cv2.solvePnPRansac` with matched `5000` iteration budget for 2D–3D absolute pose, and `cv2.estimateAffine3D` + SVD orthogonalization with matched 3D inlier threshold for 3D–3D rigid registration):

### 6D Absolute Pose Estimation (2D–3D $PnP$)

Evaluated across 140 pairs spanning outlier ratios from `33%` to `91%` on `pose6dscene` (`examples/example_absolute_pose_fitting.ipynb`), `SupeRANSAC` achieves **`0.7336`** Pose $\text{AUC}@(10^\circ, 50\text{cm})$ (`1.49°` median rotation error, `9.73 cm` median translation error) compared to `OpenCV RANSAC`'s **`0.2001`** (`5.70°` median rotation error, `44.09 cm` median translation error):

<p align="center">
  <img src="assets/absolute_pose_experiments.png" width="820"/>
</p>

| Outlier Ratio | `OpenCV RANSAC` Pose AUC | **`SupeRANSAC`** Pose AUC | `OpenCV RANSAC` Med. Rot / Trans | **`SupeRANSAC`** Med. Rot / Trans |
| :--- | :---: | :---: | :---: | :---: |
| **33% Outliers** (`0.5x`) | `0.4389` | **`0.7725`** | `3.39°` / `23.54 cm` | **`1.72°` / `8.71 cm`** |
| **50% Outliers** (`1.0x`) | `0.3401` | **`0.7276`** | `3.62°` / `31.22 cm` | **`1.53°` / `11.70 cm`** |
| **67% Outliers** (`2.0x`) | `0.2894` | **`0.7472`** | `4.05°` / `45.03 cm` | **`1.19°` / `7.57 cm`** |
| **75% Outliers** (`3.0x`) | `0.2802` | **`0.7952`** | `5.40°` / `38.50 cm` | **`1.46°` / `5.66 cm`** |
| **83% Outliers** (`5.0x`) | `0.0561` | **`0.6982`** | `9.63°` / `90.45 cm` | **`1.78°` / `10.06 cm`** |
| **88% Outliers** (`7.0x`) | `0.0855` | **`0.7557`** | `14.21°` / `85.22 cm` | **`1.55°` / `11.33 cm`** |
| **91% Outliers** (`10.0x`) | `0.0362` | **`0.7341`** | `59.94°` / `241.75 cm` | **`1.50°` / `11.07 cm`** |
| **Overall (140 pairs)** | `0.2001` | **`0.7336`** (**`+53.34 pt`**) | `5.70°` / `44.09 cm` | **`1.49°` / `9.73 cm`** |

### 6D Rigid Transformation Estimation (3D–3D $SE(3)$)

Evaluated across 100 3D–3D registration pairs spanning 4 difficulty tiers (`Tier 0` clean to `Tier 3` extreme outlier ratios), `SupeRANSAC` achieves **`0.8496`** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ compared to `OpenCV RANSAC`'s **`0.1666`** (**`+68.30 pt`**):

<p align="center">
  <img src="assets/rigid_transform_experiments.png" width="820"/>
</p>

| Metric / Difficulty Tier | `OpenCV RANSAC` | **`SupeRANSAC`** | Gain vs. `OpenCV RANSAC` |
| :--- | :---: | :---: | :---: |
| **Overall Pose $\text{AUC}@(5^\circ, 10\text{cm})$** | `0.1666` | **`0.8496`** | **`+68.30 pt`** |
| **Overall Rotation $\text{AUC}@5^\circ$** | `0.2148` | **`0.8966`** | **`+68.17 pt`** |
| **Overall Translation $\text{AUC}@10\text{cm}$** | `0.1926` | **`0.8586`** | **`+66.60 pt`** |
| **Tier 0 (Clean)** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `0.1468` | **`0.7819`** | **`+63.50 pt`** |
| **Tier 1 (Moderate)** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `0.1025` | **`0.8340`** | **`+73.14 pt`** |
| **Tier 2 (High Outliers)** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `0.4071` | **`0.9204`** | **`+51.33 pt`** |
| **Tier 3 (Extreme Outliers)** Pose $\text{AUC}@(5^\circ, 10\text{cm})$ | `0.0597` | **`0.8801`** | **`+82.04 pt`** |

### Datasets, Evaluation Protocol & Exact Reproduction (Section 4)

1. **Datasets & Evaluation Protocol**:
   - **6D Absolute Pose (2D–3D $PnP$, 140 pairs)**: Uses the calibrated 2D–3D dataset `examples/data/pose6dscene_{points,gt}.txt` with camera intrinsics `examples/data/pose6dscene.K` (following `examples/example_absolute_pose_fitting.ipynb`). Evaluates 20 deterministic trials at each of 7 outlier multipliers (`0.5x` = `33%`, `1.0x` = `50%`, `2.0x` = `67%`, `3.0x` = `75%`, `5.0x` = `83%`, `7.0x` = `88%`, `10.0x` = `91%` outliers) with $\sigma = 1.5\text{ px}$ 2D Gaussian noise. Compares `cv2.solvePnPRansac(..., iterationsCount=5000)` against `pysuperansac.estimateAbsolutePose` at threshold $3.5\text{ px}$.
   - **6D Rigid Transformation (3D–3D $SE(3)$, 100 pairs)**: Uses the 100 3D–3D registration pairs in `benchmark_data.pkl` built from `examples/data/rigid_pose_example_{points,gt}.txt` across 4 difficulty tiers (`Tier 0` clean to `Tier 3` extreme outlier ratio). Compares `cv2.estimateAffine3D` + SVD rotation orthogonalization against `pysuperansac.estimateRigidTransform`.

2. **Exact Reproduction Command (Section 4)**:
   ```bash
   source ~/ransac/.venv/bin/activate
   python benchmarks/generate_readme_plots.py --section 4
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

The testing scripts for SupeRANSAC are located in `tests/`:
- Fundamental matrix: `python tests/fundamental_matrix/tester_superansac.py --features LoMa`
- Essential matrix: `python tests/essential_matrix/tester_superansac.py --features LoMa`
- Homography matrix: `python tests/homography/tester_superansac.py --features LoMa`
- Rigid 6D transformation: `python tests/rigid_transformation/tester_superansac.py`

Supported `--features` options include `LoMa`, `RoMa`, `LightGlue`, `LoFTR`, `RootSIFT`, `SIFT`, and `ORB`.

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

### Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```bash
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf RANSAC-Tutorial-Data-EF.tar
```

Then run the notebook `examples/relative_pose_evaluation_phototourism.ipynb`.

### Evaluation on the ScanNet dataset
Download the data from the test set for relative pose estimation used in SuperGlue (~250Mb for 1500 image pairs only):
```bash
wget https://www.polybox.ethz.ch/index.php/s/lAZyxm62WUh27Zl/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

Then run the notebook `examples/relative_pose_evaluation_scannet.ipynb`.

### Evaluation on the 7Scenes dataset
Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and put it where it suits you. You can also only download one scene and later specify this scene in the dataloader constructor.

Then run the script `runners/run_7scenes.py`.

### Evaluation on the ETH3D dataset
Download the [ETH3D dataset](https://www.eth3d.net/datasets) (training split of the high-res multi-view, undistorted images + GT extrinsics & intrinsics should be enough) and put it where it suits you. The input argument `downsize_factor` can be used to downscale the images, because they can be quite large otherwise.

Then run the script `runners/run_eth3d.py`.

### Evaluation on the LaMAR dataset
Download the [CAB scene of the LaMAR dataset](https://cvg-data.inf.ethz.ch/lamar/CAB.zip), and unzip it to your favourite location. Note that we only use the images in `CAB/sessions/query_val_hololens`.

### Evaluation on the KITTI dataset
Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images and poses), and unzip them to your favourite location.

