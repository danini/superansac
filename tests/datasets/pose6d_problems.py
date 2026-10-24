"""6D pose problem sets: absolute pose (2D-3D, PnP) and rigid registration (3D-3D, SE(3)).

Real track (headline):
    7Scenes RGB-D. For each frozen 7Scenes pair and each detector, the detector's matches are lifted
    to 3D through the depth map of image 1 (and image 2 for 3D-3D). Ground truth = the dataset pose
    files. Nothing is injected or filtered: the outliers are the detector's own mismatches and depth
    noise/holes. (7Scenes depth and colour are only approximately registered, which adds a small
    bias to the GT; it affects both methods identically.)

Synthetic stress tests (reported separately, labelled as such):
    pose6dscene        : one calibrated 2D-3D scene + uniformly random 2D-3D outliers (33%-91%)
    rigid_pose_example : one real 3D-3D scan pair, subsampled, with random-rotation/translation and
                         uniformly random outliers in 4 tiers
"""

import os

import cv2
import numpy as np

from datasets.benchmark_pairs import (
    DEFAULT_DATA_DIR,
    SUPERANSAC_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    load_calibrated_pairs,
)

SEVEN_SCENES_DEPTH_K = np.array([[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]])
PNP_DETECTORS = ("LoMa", "RoMa", "DISK+LG", "RootSIFT", "SIFT", "ORB")


def _lift(depth_img, uv, K_inv):
    """Back-project pixel coordinates through a 7Scenes depth map (mm, 65535 = invalid)."""
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < depth_img.shape[1]) & (v >= 0) & (v < depth_img.shape[0])
    z = np.zeros(len(uv))
    z[ok] = depth_img[v[ok], u[ok]]
    ok &= (z > 100) & (z < 5000)
    z_m = z / 1000.0
    X = (K_inv @ np.hstack([uv, np.ones((len(uv), 1))]).T).T * z_m[:, None]
    return X, ok


def load_7scenes_pnp_problems(data_dir=DEFAULT_DATA_DIR, detectors=PNP_DETECTORS):
    pairs = [p for p in load_calibrated_pairs(data_dir, verbose=False) if p["dataset"] == "7Scenes"]
    cache = ensure_cached_matches(pairs, list(detectors), data_dir=data_dir)
    K_inv = np.linalg.inv(SEVEN_SCENES_DEPTH_K)
    problems = []
    for p in pairs:
        d1 = cv2.imread(p["depth1"], cv2.IMREAD_UNCHANGED) if p["depth1"] else None
        if d1 is None:
            continue
        for det in detectors:
            det = canonical_detector_name(det)
            corrs, scores = cache[p["cache_key"]][det]
            if len(corrs) < 20:
                continue
            X1, ok = _lift(d1, corrs[:, :2], K_inv)
            if ok.sum() < 20:
                continue
            problems.append({
                "dataset": "7Scenes RGB-D", "detector": det, "scene": p["scene"],
                "corrs": np.ascontiguousarray(np.hstack([corrs[ok, 2:4], X1[ok]]), dtype=np.float64),
                "probs": np.ascontiguousarray(scores[ok], dtype=np.float64),
                "K": p["K2"], "R_gt": p["R"], "t_gt": p["t"],
                "width": float(p["w2"]), "height": float(p["h2"]),
                "threshold_px": 3.5, "cm_scale": 100.0,
            })
    return problems


def load_7scenes_rigid_problems(data_dir=DEFAULT_DATA_DIR, detectors=PNP_DETECTORS):
    pairs = [p for p in load_calibrated_pairs(data_dir, verbose=False) if p["dataset"] == "7Scenes"]
    cache = ensure_cached_matches(pairs, list(detectors), data_dir=data_dir)
    K_inv = np.linalg.inv(SEVEN_SCENES_DEPTH_K)
    problems = []
    for p in pairs:
        d1 = cv2.imread(p["depth1"], cv2.IMREAD_UNCHANGED) if p["depth1"] else None
        d2 = cv2.imread(p["depth2"], cv2.IMREAD_UNCHANGED) if p["depth2"] else None
        if d1 is None or d2 is None:
            continue
        for det in detectors:
            det = canonical_detector_name(det)
            corrs, scores = cache[p["cache_key"]][det]
            if len(corrs) < 20:
                continue
            X1, ok1 = _lift(d1, corrs[:, :2], K_inv)
            X2, ok2 = _lift(d2, corrs[:, 2:4], K_inv)
            ok = ok1 & ok2
            if ok.sum() < 20:
                continue
            problems.append({
                "dataset": "7Scenes RGB-D", "detector": det, "scene": p["scene"],
                "corrs": np.ascontiguousarray(np.hstack([X1[ok], X2[ok]]), dtype=np.float64),
                "probs": np.ascontiguousarray(scores[ok], dtype=np.float64),
                "R_gt": p["R"], "t_gt": p["t"],
                "threshold_3d": 0.05, "cm_scale": 100.0,
            })
    return problems


# --------------------------------------------------------------------------------------
# Synthetic stress tests (clearly labelled as such in the outputs)
# --------------------------------------------------------------------------------------
def _random_rotation_matrix(rng, max_angle_deg=90.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def load_pose6dscene_pnp_problems(n_trials_per_regime=20):
    ex = os.path.join(SUPERANSAC_DIR, "examples", "data")
    pts_path, gt_path, k_path = (os.path.join(ex, f) for f in ("pose6dscene_points.txt", "pose6dscene_gt.txt", "pose6dscene.K"))
    if not all(os.path.exists(p) for p in (pts_path, gt_path, k_path)):
        return []
    corrs_gt = np.loadtxt(pts_path, dtype=np.float64)
    gt_pose = np.loadtxt(gt_path, dtype=np.float64)
    K = np.loadtxt(k_path, dtype=np.float64)
    R_gt, t_gt = gt_pose[:, :3].copy(), gt_pose[:, 3].copy()
    min_gt, max_gt = corrs_gt.min(axis=0), corrs_gt.max(axis=0)
    w = float(max(K[0, 2] * 2.0, max_gt[0] + 20.0))
    h = float(max(K[1, 2] * 2.0, max_gt[1] + 20.0))
    problems = []
    for m_idx, (mult, lbl) in enumerate(zip([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], ["33%", "50%", "67%", "75%", "83%", "88%", "91%"])):
        rng = np.random.default_rng(2026 + m_idx * 37)
        n_out = int(round(mult * len(corrs_gt)))
        for trial in range(n_trials_per_regime):
            pts = np.vstack([corrs_gt, rng.uniform(min_gt, max_gt, size=(n_out, 5))])
            pts[:, :2] += rng.normal(0.0, 1.5, size=(len(pts), 2))
            pts = pts[rng.permutation(len(pts))]
            problems.append({
                "dataset": "pose6dscene (synthetic outliers)", "detector": f"outliers {lbl}", "scene": f"pose6d_{lbl}_{trial}",
                "corrs": np.ascontiguousarray(pts, dtype=np.float64), "probs": np.zeros(0),
                "K": K, "R_gt": R_gt, "t_gt": t_gt, "width": w, "height": h,
                "threshold_px": 3.5, "cm_scale": 1.0,  # scene units are cm
            })
    return problems


def load_rigid_pose_example_problems(n_trials_per_tier=25):
    ex = os.path.join(SUPERANSAC_DIR, "examples", "data")
    pts_path, gt_path = os.path.join(ex, "rigid_pose_example_points.txt"), os.path.join(ex, "rigid_pose_example_gt.txt")
    if not (os.path.exists(pts_path) and os.path.exists(gt_path)):
        return []
    rng = np.random.default_rng(4004)
    real = np.loadtxt(pts_path, dtype=np.float64)
    T_gt = np.loadtxt(gt_path, dtype=np.float64)[:4, :]
    p1_h = np.hstack([real[:, :3], np.ones((len(real), 1))])
    res = np.linalg.norm((p1_h @ T_gt.T)[:, :3] - real[:, 3:6], axis=1)
    inliers = real[res < 0.05]
    natural_out = real[res >= 0.15]
    problems = []
    # Tier 0: the real scan pair as shipped (natural outliers only)
    problems.append({
        "dataset": "rigid_pose_example", "detector": "real scan (as shipped)", "scene": "real_scan",
        "corrs": np.ascontiguousarray(real), "probs": np.zeros(0),
        "R_gt": T_gt[:3, :3], "t_gt": T_gt[:3, 3], "threshold_3d": 0.05, "cm_scale": 100.0,
    })
    tiers = [("synthetic 70-85% outliers", 35, 90, 150, 400, 45.0), ("synthetic 85-92% outliers", 30, 80, 300, 750, 90.0)]
    for name, n_in_lo, n_in_hi, n_out_lo, n_out_hi, max_rot in tiers:
        for i in range(n_trials_per_tier):
            n_in, n_out = int(rng.integers(n_in_lo, n_in_hi)), int(rng.integers(n_out_lo, n_out_hi))
            inl = inliers[rng.choice(len(inliers), size=min(n_in, len(inliers)), replace=False)].copy()
            n_nat = min(len(natural_out), n_out // 2)
            nat = natural_out[rng.choice(len(natural_out), size=n_nat, replace=False)]
            mn, mx = real.min(axis=0) - 0.5, real.max(axis=0) + 0.5
            uni = rng.uniform(mn, mx, size=(n_out - n_nat, 6))
            corrs = np.vstack([inl, nat, uni])
            R_d = _random_rotation_matrix(rng, max_rot)
            t_d = rng.uniform(-1.0, 1.0, size=3)
            corrs[:, 3:6] = corrs[:, 3:6] @ R_d.T + t_d
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R_d, t_d
            T_new = T @ T_gt
            corrs = corrs[rng.permutation(len(corrs))]
            problems.append({
                "dataset": "rigid_pose_example", "detector": name, "scene": f"{name}_{i:02d}",
                "corrs": np.ascontiguousarray(corrs), "probs": np.zeros(0),
                "R_gt": T_new[:3, :3], "t_gt": T_new[:3, 3], "threshold_3d": 0.05, "cm_scale": 100.0,
            })
    return problems

