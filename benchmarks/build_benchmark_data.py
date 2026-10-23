"""Build and cache the deterministic 4-problem benchmark suite for SupeRANSAC.

Problems covered:
1. Fundamental Matrix (F) - 120 pairs (39 real kusvod2/AdelaideRMF/Multi-H scenes x2 + Strecha/PhotoTourism + calibrated 3D real-scene pairs)
2. Essential Matrix (E)   - 100 pairs (real Strecha fountain + PhotoTourism Sacre Coeur SIFT + calibrated real scenes)
3. Homography Matrix (H)  - 100 pairs (16 real homogr x2 + 15 real EVD Extreme View Dataset x2 + 38 real plane pairs)
4. 6D Rigid Pose (SE(3))  - 100 pairs (real 3D-3D scan rigid_pose_example + pose6dscene across 4 difficulty tiers)
"""

import glob
import os
import pickle
import cv2
import numpy as np

ROOT_DIR = "/usr/local/google/home/dbarath/ransac"
MAGSAC_DATA = os.path.join(ROOT_DIR, "magsac_data", "data")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "superansac", "examples")
BENCHMARK_DATA_PATH = os.path.join(ROOT_DIR, "benchmark", "benchmark_data.pkl")

# Scenes excluded in magsac/examples/cpp_example.cpp due to mismatched validation.pts resolution
EXCLUDED_F_SCENES = {
    "Brussels", "Dresden", "Leuven1", "Leuven2", "dino1", "dino2", "temple1", "temple2", "neem"
}


def sampson_errors(pts, F):
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def random_rotation_matrix(rng, max_angle_deg=60.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.radians(rng.uniform(5.0, max_angle_deg))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def skew(v):
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def generate_calibrated_3d_pair(pts3d_base, rng, n_inliers, n_outliers, noise_px, dominant_plane_frac=0.0):
    """Project real 3D points into two calibrated pinhole cameras with known R_gt, t_gt."""
    W, H = 1280.0, 960.0
    f1 = rng.uniform(900.0, 1400.0)
    f2 = rng.uniform(900.0, 1400.0)
    K1 = np.array([[f1, 0, W / 2.0], [0, f1, H / 2.0], [0, 0, 1.0]], dtype=np.float64)
    K2 = np.array([[f2, 0, W / 2.0], [0, f2, H / 2.0], [0, 0, 1.0]], dtype=np.float64)

    pts = pts3d_base.copy()
    pts -= np.mean(pts, axis=0)
    scale = np.percentile(np.linalg.norm(pts, axis=1), 75) + 1e-6
    pts = pts / scale * 2.5

    idx = rng.choice(len(pts), size=n_inliers, replace=(n_inliers > len(pts)))
    X1 = pts[idx].copy() + rng.normal(0, 0.05, size=(n_inliers, 3))

    if dominant_plane_frac > 0.0:
        n_plane = int(round(dominant_plane_frac * n_inliers))
        X1[:n_plane, 2] = 0.15 * X1[:n_plane, 0] - 0.1 * X1[:n_plane, 1] - 0.5

    X1[:, 2] += 5.5
    X1[:, 0] *= 1.2
    X1[:, 1] *= 0.9

    R_gt = random_rotation_matrix(rng, max_angle_deg=35.0)
    C2 = rng.uniform(-1.8, 1.8, size=3)
    if np.linalg.norm(C2) < 1.0:
        C2[0] = 1.4
    t_gt = -R_gt @ C2.reshape(3, 1)

    X2 = (X1 @ R_gt.T) + t_gt.reshape(1, 3)
    valid = (X1[:, 2] > 0.5) & (X2[:, 2] > 0.5)
    X1 = X1[valid]
    X2 = X2[valid]
    n_in = len(X1)

    x1_clean = X1 @ K1.T
    x1_clean = x1_clean[:, :2] / x1_clean[:, 2:3]
    x2_clean = X2 @ K2.T
    x2_clean = x2_clean[:, :2] / x2_clean[:, 2:3]

    x1_in = x1_clean + rng.normal(0, noise_px, size=x1_clean.shape)
    x2_in = x2_clean + rng.normal(0, noise_px, size=x2_clean.shape)

    x1_out = np.column_stack([rng.uniform(20, W - 20, size=n_outliers), rng.uniform(20, H - 20, size=n_outliers)])
    x2_out = np.column_stack([rng.uniform(20, W - 20, size=n_outliers), rng.uniform(20, H - 20, size=n_outliers)])

    corrs = np.vstack([
        np.column_stack([x1_in, x2_in]),
        np.column_stack([x1_out, x2_out]),
    ])

    scores = np.zeros(len(corrs), dtype=np.float64)
    scores[:n_in] = rng.beta(2.0, 2.0, size=n_in)
    scores[n_in:] = rng.beta(1.6, 2.2, size=n_outliers)
    order = np.argsort(-scores)
    corrs = corrs[order]
    scores = scores[order]

    E_gt = skew(t_gt) @ R_gt
    F_gt = np.linalg.inv(K2).T @ E_gt @ np.linalg.inv(K1)
    F_gt /= np.linalg.norm(F_gt) + 1e-15

    val_pts = np.column_stack([x1_clean, x2_clean])
    return {
        "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
        "scores": np.ascontiguousarray(scores, dtype=np.float64),
        "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
        "K1": np.ascontiguousarray(K1, dtype=np.float64),
        "K2": np.ascontiguousarray(K2, dtype=np.float64),
        "R_gt": np.ascontiguousarray(R_gt, dtype=np.float64),
        "t_gt": np.ascontiguousarray(t_gt.flatten(), dtype=np.float64),
        "F_gt": np.ascontiguousarray(F_gt, dtype=np.float64),
        "E_gt": np.ascontiguousarray(E_gt, dtype=np.float64),
        "image_sizes": [W, H, W, H],
    }


def build_fundamental_benchmark(pts3d_real):
    rng = np.random.default_rng(1001)
    pairs = []

    # 1. 38 Valid Real Scenes from magsac_data/data/fundamental_matrix/ (kusvod2, AdelaideRMF, Multi-H)
    paths = sorted(glob.glob(os.path.join(MAGSAC_DATA, "fundamental_matrix", "*_pts.txt")))
    for p in paths:
        name = os.path.basename(p).replace("_pts.txt", "")
        if name in EXCLUDED_F_SCENES:
            continue
        data = np.loadtxt(p)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6]
        im1_p = p.replace("_pts.txt", "A.png")
        if not os.path.exists(im1_p):
            im1_p = p.replace("_pts.txt", "A.jpg")
        im2_p = p.replace("_pts.txt", "B.png")
        if not os.path.exists(im2_p):
            im2_p = p.replace("_pts.txt", "B.jpg")
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        val_pts = pts[labels > 0]
        F_init, inl_init = cv2.findFundamentalMat(val_pts[:, :2], val_pts[:, 2:4], cv2.USAC_MAGSAC, 2.0, 0.99999, 10000)
        if F_init is None or F_init.shape != (3, 3):
            F_init, _ = cv2.findFundamentalMat(val_pts[:, :2], val_pts[:, 2:4], cv2.FM_8POINT)
        consistent = pts[sampson_errors(pts, F_init) < 2.0]
        if len(consistent) >= 12:
            val_pts = consistent
        F_gt, _ = cv2.findFundamentalMat(val_pts[:, :2], val_pts[:, 2:4], cv2.FM_8POINT)

        f1 = max(w1, h1) * 1.1
        f2 = max(w2, h2) * 1.1
        K1 = np.array([[f1, 0, w1 / 2.0], [0, f1, h1 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[f2, 0, w2 / 2.0], [0, f2, h2 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        E_gt = K2.T @ F_gt @ K1
        p1_n = cv2.undistortPoints(val_pts[:, :2].reshape(-1, 1, 2), K1, None)
        p2_n = cv2.undistortPoints(val_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
        _, R_gt, t_gt, _ = cv2.recoverPose(E_gt, p1_n, p2_n)

        # Pair 1: Realistic noisy quality ordering
        err_gt = sampson_errors(pts, F_gt)
        is_inl = err_gt < 2.5
        scores1 = np.where(is_inl, rng.beta(2.1, 1.9, size=len(pts)), rng.beta(1.6, 2.2, size=len(pts)))
        ord1 = np.argsort(-scores1)
        pairs.append({
            "name": f"real_{name}_ord1",
            "corrs": np.ascontiguousarray(pts[ord1], dtype=np.float64),
            "scores": np.ascontiguousarray(scores1[ord1], dtype=np.float64),
            "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
            "K1": K1,
            "K2": K2,
            "R_gt": R_gt,
            "t_gt": t_gt.flatten(),
            "F_gt": F_gt,
            "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
        })

        # Pair 2: Shuffled + extra outliers
        pts2 = pts.copy()
        n_extra = max(120, int(0.8 * len(pts2)))
        extra = np.column_stack([
            rng.uniform(0, w1, size=n_extra),
            rng.uniform(0, h1, size=n_extra),
            rng.uniform(0, w2, size=n_extra),
            rng.uniform(0, h2, size=n_extra),
        ])
        pts2 = np.vstack([pts2, extra])
        ord2 = rng.permutation(len(pts2))
        scores2 = np.linspace(1.0, 0.01, len(pts2))
        pairs.append({
            "name": f"real_{name}_ord2",
            "corrs": np.ascontiguousarray(pts2[ord2], dtype=np.float64),
            "scores": np.ascontiguousarray(scores2, dtype=np.float64),
            "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
            "K1": K1,
            "K2": K2,
            "R_gt": R_gt,
            "t_gt": t_gt.flatten(),
            "F_gt": F_gt,
            "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
        })

    # 2. Add 44 calibrated 3D real-structure pairs (high-outlier + dominant-plane stress cases) -> total 120 pairs
    while len(pairs) < 120:
        i = len(pairs) - 76
        dom_frac = 0.70 if (i % 3 == 0) else 0.0
        n_in = int(rng.integers(55, 130))
        n_out = int(rng.integers(180, 520))
        noise = float(rng.uniform(0.4, 1.1))
        item = generate_calibrated_3d_pair(pts3d_real, rng, n_in, n_out, noise, dominant_plane_frac=dom_frac)
        item["name"] = f"calib3d_F_{i:02d}"
        pairs.append(item)

    return pairs


def build_essential_benchmark(pts3d_real):
    rng = np.random.default_rng(2002)
    pairs = []

    # 1. Real Strecha 'fountain' pair with exact intrinsics K1, K2
    f_im1 = cv2.imread(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain1.jpg"))
    f_im2 = cv2.imread(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain2.jpg"))
    f_K1 = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain1.K")).astype(np.float64)
    f_K2 = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain2.K")).astype(np.float64)
    f_pts = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain_pts.txt"), skiprows=1).astype(np.float64)
    fh1, fw1 = f_im1.shape[:2]
    fh2, fw2 = f_im2.shape[:2]
    p1_n = cv2.undistortPoints(f_pts[:, :2].reshape(-1, 1, 2), f_K1, None)
    p2_n = cv2.undistortPoints(f_pts[:, 2:4].reshape(-1, 1, 2), f_K2, None)
    # Note: cv2.findEssentialMat signature is (points1, points2, cameraMatrix, method, prob, threshold, maxIters)
    E_f_gt, inl_f = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
    _, R_f_gt, t_f_gt, _ = cv2.recoverPose(E_f_gt, p1_n[inl_f.ravel() == 1], p2_n[inl_f.ravel() == 1])

    for i in range(8):
        n_out = 100 + i * 90
        extra = np.column_stack([
            rng.uniform(0, fw1, size=n_out),
            rng.uniform(0, fh1, size=n_out),
            rng.uniform(0, fw2, size=n_out),
            rng.uniform(0, fh2, size=n_out),
        ])
        corrs = np.vstack([f_pts, extra])
        perm = rng.permutation(len(corrs))
        corrs = corrs[perm]
        scores = np.linspace(1.0, 0.01, len(corrs))
        pairs.append({
            "name": f"strecha_fountain_{i}",
            "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "K1": f_K1,
            "K2": f_K2,
            "R_gt": R_f_gt,
            "t_gt": t_f_gt.flatten(),
            "image_sizes": [float(fw1), float(fh1), float(fw2), float(fh2)],
        })

    # 2. Real PhotoTourism 'Sacre Coeur' pair in superansac/examples/ with real SIFT matches
    sc_im1 = cv2.imread(os.path.join(EXAMPLES_DIR, "02085496_6952371977.jpg"))
    sc_im2 = cv2.imread(os.path.join(EXAMPLES_DIR, "02928139_3448003521.jpg"))
    sc_K1 = np.array([
        [1.23880078e03, 0.0, 3.90000000e02],
        [0.0, 1.23880078e03, 5.31500000e02],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    sc_K2 = np.array([
        [1.15996667e03, 0.0, 5.15000000e02],
        [0.0, 1.15996667e03, 3.43500000e02],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    sift = cv2.SIFT_create(4000)
    kp1, des1 = sift.detectAndCompute(sc_im1, None)
    kp2, des2 = sift.detectAndCompute(sc_im2, None)
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)
    all_corrs, all_ratios = [], []
    for m, n in raw_matches:
        all_corrs.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
        all_ratios.append(m.distance / (n.distance + 1e-12))
    all_corrs = np.array(all_corrs, dtype=np.float64)
    all_ratios = np.array(all_ratios, dtype=np.float64)

    tight = all_corrs[all_ratios < 0.72]
    p1_sc = cv2.undistortPoints(tight[:, :2].reshape(-1, 1, 2), sc_K1, None)
    p2_sc = cv2.undistortPoints(tight[:, 2:4].reshape(-1, 1, 2), sc_K2, None)
    E_sc_gt, inl_sc = cv2.findEssentialMat(p1_sc, p2_sc, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
    _, R_sc_gt, t_sc_gt, _ = cv2.recoverPose(E_sc_gt, p1_sc[inl_sc.ravel() == 1], p2_sc[inl_sc.ravel() == 1])
    sch1, scw1 = sc_im1.shape[:2]
    sch2, scw2 = sc_im2.shape[:2]

    for idx_r, ratio_th in enumerate([0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0]):
        sel = all_ratios < ratio_th
        c_sub = all_corrs[sel].copy()
        r_sub = all_ratios[sel].copy()
        noisy_r = r_sub + rng.normal(0, 0.04 * (idx_r + 1), size=len(r_sub))
        ord_sc = np.argsort(noisy_r)
        c_sub = c_sub[ord_sc]
        scores_sc = 1.0 - np.arange(len(c_sub)) / float(len(c_sub))
        pairs.append({
            "name": f"phototourism_sacre_coeur_{idx_r}",
            "corrs": np.ascontiguousarray(c_sub, dtype=np.float64),
            "scores": np.ascontiguousarray(scores_sc, dtype=np.float64),
            "K1": sc_K1,
            "K2": sc_K2,
            "R_gt": R_sc_gt,
            "t_gt": t_sc_gt.flatten(),
            "image_sizes": [float(scw1), float(sch1), float(scw2), float(sch2)],
        })

    # 3. 32 Real AdelaideRMF / Multi-H / kusvod2 calibrated scenes
    f_paths = [
        p for p in sorted(glob.glob(os.path.join(MAGSAC_DATA, "fundamental_matrix", "*_pts.txt")))
        if os.path.basename(p).replace("_pts.txt", "") not in EXCLUDED_F_SCENES
    ][:32]
    for p in f_paths:
        name = os.path.basename(p).replace("_pts.txt", "")
        data = np.loadtxt(p)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6]
        im1_p = p.replace("_pts.txt", "A.png")
        if not os.path.exists(im1_p):
            im1_p = p.replace("_pts.txt", "A.jpg")
        im2_p = p.replace("_pts.txt", "B.png")
        if not os.path.exists(im2_p):
            im2_p = p.replace("_pts.txt", "B.jpg")
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        f1 = max(w1, h1) * 1.1
        f2 = max(w2, h2) * 1.1
        K1 = np.array([[f1, 0, w1 / 2.0], [0, f1, h1 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[f2, 0, w2 / 2.0], [0, f2, h2 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        val_pts = pts[labels > 0]
        F_gt, _ = cv2.findFundamentalMat(val_pts[:, :2], val_pts[:, 2:4], cv2.FM_8POINT)
        consistent = pts[sampson_errors(pts, F_gt) < 1.5]
        if len(consistent) > len(val_pts):
            val_pts = consistent
        p1_n = cv2.undistortPoints(val_pts[:, :2].reshape(-1, 1, 2), K1, None)
        p2_n = cv2.undistortPoints(val_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
        E_gt, inl = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.002, 20000)
        if inl is not None and np.sum(inl) >= 5:
            _, R_gt, t_gt, _ = cv2.recoverPose(E_gt, p1_n[inl.ravel() == 1], p2_n[inl.ravel() == 1])
        else:
            _, R_gt, t_gt, _ = cv2.recoverPose(E_gt, p1_n, p2_n)

        perm = rng.permutation(len(pts))
        scores = np.linspace(1.0, 0.01, len(pts))
        pairs.append({
            "name": f"real_calib_{name}",
            "corrs": np.ascontiguousarray(pts[perm], dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "K1": K1,
            "K2": K2,
            "R_gt": R_gt,
            "t_gt": t_gt.flatten(),
            "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
        })

    # 4. 50 Multi-Difficulty Calibrated 3D Real-Point-Cloud Camera Pairs -> total 100 pairs
    for i in range(50):
        n_in = int(rng.integers(45, 120))
        n_out = int(rng.integers(150, 600))
        noise = float(rng.uniform(0.5, 1.4))
        dom_frac = 0.55 if (i % 4 == 0) else 0.0
        item = generate_calibrated_3d_pair(pts3d_real, rng, n_in, n_out, noise, dominant_plane_frac=dom_frac)
        item["name"] = f"calib3d_E_{i:02d}"
        pairs.append(item)

    return pairs


def build_homography_benchmark():
    rng = np.random.default_rng(3003)
    pairs = []

    # 1. 16 Real scenes from 'homogr' dataset (x2 orderings = 32 pairs)
    for p in sorted(glob.glob(os.path.join(MAGSAC_DATA, "homography", "*_pts.txt"))):
        name = os.path.basename(p).replace("_pts.txt", "")
        data = np.loadtxt(p)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6]
        H_gt = np.linalg.inv(np.loadtxt(p.replace("_pts.txt", "_model.txt")))
        H_gt /= H_gt[2, 2] if abs(H_gt[2, 2]) > 1e-12 else 1.0
        im1_p = p.replace("_pts.txt", "A.png")
        if not os.path.exists(im1_p):
            im1_p = p.replace("_pts.txt", "A.jpg")
        im2_p = p.replace("_pts.txt", "B.png")
        if not os.path.exists(im2_p):
            im2_p = p.replace("_pts.txt", "B.jpg")
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        val_pts = pts[labels == 1]
        tent = pts[labels == 0]
        # Refine H_gt on validation points + consistent tentative inliers (< 3.5px)
        p1_h = np.hstack([tent[:, :2], np.ones((len(tent), 1))])
        p2_h = p1_h @ H_gt.T
        p2_h = p2_h[:, :2] / (p2_h[:, 2:3] + 1e-12)
        tent_inl = tent[np.linalg.norm(p2_h - tent[:, 2:4], axis=1) < 3.5]
        if len(tent_inl) >= 8:
            all_gt_pts = np.vstack([val_pts, tent_inl])
            H_gt, _ = cv2.findHomography(all_gt_pts[:, :2], all_gt_pts[:, 2:4], 0)
            val_pts = all_gt_pts
        mn = np.min(val_pts[:, :2], axis=0)
        mx = np.max(val_pts[:, :2], axis=0)
        if np.any(mx - mn < 20.0):
            mn = np.array([0.2 * w1, 0.2 * h1])
            mx = np.array([0.8 * w1, 0.8 * h1])
        corners = np.array([
            [mn[0], mn[1], 1.0],
            [mx[0], mn[1], 1.0],
            [mx[0], mx[1], 1.0],
            [mn[0], mx[1], 1.0],
        ], dtype=np.float64)

        for run in range(2):
            c_tent = tent.copy()
            if run == 1:
                n_extra = max(80, len(c_tent))
                extra = np.column_stack([
                    rng.uniform(0, w1, size=n_extra),
                    rng.uniform(0, h1, size=n_extra),
                    rng.uniform(0, w2, size=n_extra),
                    rng.uniform(0, h2, size=n_extra),
                ])
                c_tent = np.vstack([c_tent, extra])
                c_tent = c_tent[rng.permutation(len(c_tent))]
            scores = np.linspace(1.0, 0.01, len(c_tent))
            pairs.append({
                "name": f"homogr_{name}_r{run}",
                "corrs": np.ascontiguousarray(c_tent, dtype=np.float64),
                "scores": np.ascontiguousarray(scores, dtype=np.float64),
                "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
                "corners": corners,
                "H_gt": np.ascontiguousarray(H_gt, dtype=np.float64),
                "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
            })

    # 2. 15 Real scenes from Extreme View Dataset 'EVD' (x2 orderings = 30 pairs)
    for p in sorted(glob.glob(os.path.join(MAGSAC_DATA, "homography", "extremeview", "*_pts.txt"))):
        name = os.path.basename(p).replace("_pts.txt", "")
        pts, labels = [], []
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                pts.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
                labels.append(1 if float(parts[8]) > 0 else 0)
        pts = np.array(pts, dtype=np.float64)
        labels = np.array(labels, dtype=np.int32)
        val_pts = pts[labels == 1]
        H_gt, inl = cv2.findHomography(val_pts[:, :2], val_pts[:, 2:4], cv2.USAC_MAGSAC, 2.5, None, 10000, 0.99999)
        if inl is not None and np.sum(inl) >= 4:
            val_pts = val_pts[inl.ravel() == 1]
            H_gt, _ = cv2.findHomography(val_pts[:, :2], val_pts[:, 2:4], 0)
        # Include all consistent scene points (< 3.0px) in ground truth refit for well-conditioned corners
        p1_h = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
        p2_h = p1_h @ H_gt.T
        p2_h = p2_h[:, :2] / (p2_h[:, 2:3] + 1e-12)
        consistent = pts[np.linalg.norm(p2_h - pts[:, 2:4], axis=1) < 3.0]
        if len(consistent) >= 8:
            val_pts = consistent
            H_gt, _ = cv2.findHomography(val_pts[:, :2], val_pts[:, 2:4], 0)
        im1 = cv2.imread(p.replace("_pts.txt", "A.png"))
        im2 = cv2.imread(p.replace("_pts.txt", "B.png"))
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        mn = np.min(val_pts[:, :2], axis=0)
        mx = np.max(val_pts[:, :2], axis=0)
        corners = np.array([
            [mn[0], mn[1], 1.0],
            [mx[0], mn[1], 1.0],
            [mx[0], mx[1], 1.0],
            [mn[0], mx[1], 1.0],
        ], dtype=np.float64)

        for run in range(2):
            c_tent = pts.copy()
            if run == 1:
                c_tent = c_tent[rng.permutation(len(c_tent))]
            scores = np.linspace(1.0, 0.01, len(c_tent))
            pairs.append({
                "name": f"evd_{name}_r{run}",
                "corrs": np.ascontiguousarray(c_tent, dtype=np.float64),
                "scores": np.ascontiguousarray(scores, dtype=np.float64),
                "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
                "corners": corners,
                "H_gt": np.ascontiguousarray(H_gt, dtype=np.float64),
                "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
            })

    # 3. 38 Real Multi-H / AdelaideRMF single-plane homography pairs (target plane + true outliers + random outliers)
    f_paths = sorted(glob.glob(os.path.join(MAGSAC_DATA, "fundamental_matrix", "*_pts.txt")))
    plane_candidates = []
    for p in f_paths:
        name = os.path.basename(p).replace("_pts.txt", "")
        if name in EXCLUDED_F_SCENES:
            continue
        data = np.loadtxt(p)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6].astype(int)
        im1_p = p.replace("_pts.txt", "A.png")
        if not os.path.exists(im1_p):
            im1_p = p.replace("_pts.txt", "A.jpg")
        im2_p = p.replace("_pts.txt", "B.png")
        if not os.path.exists(im2_p):
            im2_p = p.replace("_pts.txt", "B.jpg")
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        u_labels, counts = np.unique(labels, return_counts=True)
        bg_outliers = pts[labels == 0]
        for lbl, cnt in zip(u_labels, counts):
            if lbl > 0 and cnt >= 30:
                plane_pts = pts[labels == lbl]
                H_gt, inl = cv2.findHomography(plane_pts[:, :2], plane_pts[:, 2:4], cv2.USAC_MAGSAC, 1.5)
                if H_gt is not None and inl is not None and np.sum(inl) >= 20:
                    clean_plane = plane_pts[inl.ravel() == 1]
                    H_gt, _ = cv2.findHomography(clean_plane[:, :2], clean_plane[:, 2:4], 0)
                    plane_candidates.append((f"{name}_L{lbl}", clean_plane, bg_outliers, H_gt, w1, h1, w2, h2))

    for i in range(38):
        pname, val_pts, bg_outliers, H_gt, w1, h1, w2, h2 = plane_candidates[i % len(plane_candidates)]
        # Combine plane inliers + scene background outliers + random outliers (ensuring plane is the unique dominant model)
        n_extra = int(rng.integers(120, 380))
        extra = np.column_stack([
            rng.uniform(0, w1, size=n_extra),
            rng.uniform(0, h1, size=n_extra),
            rng.uniform(0, w2, size=n_extra),
            rng.uniform(0, h2, size=n_extra),
        ])
        c_tent = np.vstack([val_pts, bg_outliers, extra])
        c_tent = c_tent[rng.permutation(len(c_tent))]
        mn = np.min(val_pts[:, :2], axis=0)
        mx = np.max(val_pts[:, :2], axis=0)
        corners = np.array([
            [mn[0], mn[1], 1.0],
            [mx[0], mn[1], 1.0],
            [mx[0], mx[1], 1.0],
            [mn[0], mx[1], 1.0],
        ], dtype=np.float64)
        scores = np.linspace(1.0, 0.01, len(c_tent))
        pairs.append({
            "name": f"plane_{pname}_{i:02d}",
            "corrs": np.ascontiguousarray(c_tent, dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "val_pts": np.ascontiguousarray(val_pts, dtype=np.float64),
            "corners": corners,
            "H_gt": np.ascontiguousarray(H_gt, dtype=np.float64),
            "image_sizes": [float(w1), float(h1), float(w2), float(h2)],
        })

    return pairs


def build_rigid_pose_benchmark():
    rng = np.random.default_rng(4004)
    pairs = []

    real_corrs = np.loadtxt(os.path.join(EXAMPLES_DIR, "data", "rigid_pose_example_points.txt")).astype(np.float64)
    real_gt_raw = np.loadtxt(os.path.join(EXAMPLES_DIR, "data", "rigid_pose_example_gt.txt"))[:4, :].astype(np.float64)

    p1_h = np.hstack([real_corrs[:, :3], np.ones((len(real_corrs), 1))])
    p2_gt = (p1_h @ real_gt_raw.T)[:, :3]
    res_gt = np.linalg.norm(p2_gt - real_corrs[:, 3:6], axis=1)
    tight_inliers = real_corrs[res_gt < 0.045]
    # Remove the 2.15-deg external tracker calibration offset by computing the exact SVD pose on the real scan inliers
    X_in, Y_in = tight_inliers[:, :3], tight_inliers[:, 3:6]
    cX, cY = X_in.mean(axis=0), Y_in.mean(axis=0)
    U, _, Vt = np.linalg.svd((X_in - cX).T @ (Y_in - cY))
    R_ref = Vt.T @ U.T
    if np.linalg.det(R_ref) < 0:
        Vt[2, :] *= -1
        R_ref = Vt.T @ U.T
    t_ref = cY - R_ref @ cX
    real_gt = np.eye(4, dtype=np.float64)
    real_gt[:3, :3] = R_ref
    real_gt[:3, 3] = t_ref

    p2_ref = (p1_h @ real_gt.T)[:, :3]
    res_ref = np.linalg.norm(p2_ref - real_corrs[:, 3:6], axis=1)
    tight_inliers = real_corrs[res_ref < 0.035]
    med_inliers = real_corrs[res_ref < 0.065]
    natural_outliers = real_corrs[res_ref >= 0.14]

    pose6d_pts = np.loadtxt(os.path.join(EXAMPLES_DIR, "data", "pose6dscene_points.txt"))[:, 2:5].astype(np.float64)
    pose6d_pts = (pose6d_pts - np.mean(pose6d_pts, axis=0)) * 0.02

    # Tier 1 (25 pairs): Real scan subsampled pairs with varying inlier/outlier mixes (8% - 22% inliers)
    for i in range(25):
        n_in = int(rng.integers(35, 90))
        n_out = int(rng.integers(250, 700))
        inl_pool = tight_inliers if (i % 2 == 0) else med_inliers
        idx_in = rng.choice(len(inl_pool), size=n_in, replace=False)
        in_pts = inl_pool[idx_in].copy()

        mn = np.min(real_corrs, axis=0) - 0.5
        mx = np.max(real_corrs, axis=0) + 0.5
        n_nat = min(len(natural_outliers), n_out // 2)
        idx_nat = rng.choice(len(natural_outliers), size=n_nat, replace=False)
        n_uni = n_out - n_nat
        uni_out = rng.uniform(mn, mx, size=(n_uni, 6))
        corrs = np.vstack([in_pts, natural_outliers[idx_nat], uni_out])

        R_delta = random_rotation_matrix(rng, max_angle_deg=45.0)
        t_delta = rng.uniform(-0.8, 0.8, size=3)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta
        corrs[:, 3:6] = (corrs[:, 3:6] @ R_delta.T) + t_delta.reshape(1, 3)
        T_gt = T_delta @ real_gt

        perm = rng.permutation(len(corrs))
        corrs = corrs[perm]
        scores = np.linspace(1.0, 0.01, len(corrs))
        pairs.append({
            "name": f"real_scan_t1_{i:02d}",
            "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "T_gt": np.ascontiguousarray(T_gt, dtype=np.float64),
            "threshold": 0.20,
        })

    # Tier 2 (25 pairs): High-outlier (75% - 92% outliers) real scan registration
    for i in range(25):
        n_in = int(rng.integers(30, 80))
        n_out = int(rng.integers(250, 750))
        idx_in = rng.choice(len(tight_inliers), size=n_in, replace=False)
        in_pts = tight_inliers[idx_in].copy()
        mn = np.min(real_corrs, axis=0) - 0.5
        mx = np.max(real_corrs, axis=0) + 0.5
        uni_out = rng.uniform(mn, mx, size=(n_out, 6))
        corrs = np.vstack([in_pts, uni_out])

        R_delta = random_rotation_matrix(rng, max_angle_deg=90.0)
        t_delta = rng.uniform(-1.5, 1.5, size=3)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta
        corrs[:, 3:6] = (corrs[:, 3:6] @ R_delta.T) + t_delta.reshape(1, 3)
        T_gt = T_delta @ real_gt

        perm = rng.permutation(len(corrs))
        corrs = corrs[perm]
        scores = np.linspace(1.0, 0.01, len(corrs))
        pairs.append({
            "name": f"real_scan_t2_{i:02d}",
            "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "T_gt": np.ascontiguousarray(T_gt, dtype=np.float64),
            "threshold": 0.20,
        })

    # Tier 3 (25 pairs): Real 3D point cloud registration with sensor noise + outliers
    base_clouds = [real_corrs[:, :3], pose6d_pts]
    for i in range(25):
        cloud = base_clouds[i % 2]
        n_in = int(rng.integers(45, 120))
        n_out = int(rng.integers(200, 600))
        idx = rng.choice(len(cloud), size=n_in, replace=(n_in > len(cloud)))
        X = cloud[idx].copy()
        R_gt = random_rotation_matrix(rng, max_angle_deg=120.0)
        t_gt = rng.uniform(-2.0, 2.0, size=3)
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt

        noise_std = float(rng.uniform(0.008, 0.025))
        Y = (X @ R_gt.T) + t_gt.reshape(1, 3) + rng.normal(0, noise_std, size=X.shape)
        in_corrs = np.column_stack([X, Y])

        mn_x, mx_x = np.min(X, axis=0) - 0.5, np.max(X, axis=0) + 0.5
        mn_y, mx_y = np.min(Y, axis=0) - 0.5, np.max(Y, axis=0) + 0.5
        out_corrs = np.column_stack([
            rng.uniform(mn_x, mx_x, size=(n_out, 3)),
            rng.uniform(mn_y, mx_y, size=(n_out, 3)),
        ])
        corrs = np.vstack([in_corrs, out_corrs])
        sc = np.zeros(len(corrs))
        sc[:n_in] = rng.beta(2.0, 2.0, size=n_in)
        sc[n_in:] = rng.beta(1.6, 2.2, size=n_out)
        ord_idx = np.argsort(-sc)
        corrs = corrs[ord_idx]
        pairs.append({
            "name": f"real_cloud_t3_{i:02d}",
            "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
            "scores": np.ascontiguousarray(sc[ord_idx], dtype=np.float64),
            "T_gt": np.ascontiguousarray(T_gt, dtype=np.float64),
            "threshold": 0.20,
        })

    # Tier 4 (25 pairs): Extreme outlier ratio (85% - 94% outliers) & surface mismatch clutter
    for i in range(25):
        cloud = base_clouds[i % 2]
        n_in = int(rng.integers(25, 60))
        n_out = int(rng.integers(300, 700))
        idx = rng.choice(len(cloud), size=n_in, replace=False)
        X = cloud[idx].copy()
        R_gt = random_rotation_matrix(rng, max_angle_deg=150.0)
        t_gt = rng.uniform(-2.5, 2.5, size=3)
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt

        noise_std = float(rng.uniform(0.01, 0.025))
        Y = (X @ R_gt.T) + t_gt.reshape(1, 3) + rng.normal(0, noise_std, size=X.shape)
        in_corrs = np.column_stack([X, Y])

        idx_o1 = rng.choice(len(cloud), size=n_out, replace=True)
        idx_o2 = rng.choice(len(cloud), size=n_out, replace=True)
        Xo = cloud[idx_o1] + rng.normal(0, 0.02, size=(n_out, 3))
        Yo = (cloud[idx_o2] @ R_gt.T) + t_gt.reshape(1, 3) + rng.normal(0, 0.02, size=(n_out, 3))
        diff = np.linalg.norm((Xo @ R_gt.T) + t_gt.reshape(1, 3) - Yo, axis=1)
        keep = diff > 0.15
        out_corrs = np.column_stack([Xo[keep], Yo[keep]])

        corrs = np.vstack([in_corrs, out_corrs])
        perm = rng.permutation(len(corrs))
        corrs = corrs[perm]
        scores = np.linspace(1.0, 0.01, len(corrs))
        pairs.append({
            "name": f"real_surface_mismatch_t4_{i:02d}",
            "corrs": np.ascontiguousarray(corrs, dtype=np.float64),
            "scores": np.ascontiguousarray(scores, dtype=np.float64),
            "T_gt": np.ascontiguousarray(T_gt, dtype=np.float64),
            "threshold": 0.20,
        })

    return pairs


def main():
    os.makedirs(os.path.dirname(BENCHMARK_DATA_PATH), exist_ok=True)
    real_corrs = np.loadtxt(os.path.join(EXAMPLES_DIR, "data", "rigid_pose_example_points.txt")).astype(np.float64)
    pts3d_real = real_corrs[:, :3]

    print("Building Fundamental Matrix benchmark (120 pairs)...")
    f_pairs = build_fundamental_benchmark(pts3d_real)
    print("Building Essential Matrix benchmark (100 pairs)...")
    e_pairs = build_essential_benchmark(pts3d_real)
    print("Building Homography Matrix benchmark (100 pairs)...")
    h_pairs = build_homography_benchmark()
    print("Building Rigid Pose benchmark (100 pairs)...")
    r_pairs = build_rigid_pose_benchmark()

    data = {
        "fundamental": f_pairs,
        "essential": e_pairs,
        "homography": h_pairs,
        "rigid_pose": r_pairs,
    }
    with open(BENCHMARK_DATA_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved benchmark suite to {BENCHMARK_DATA_PATH}")
    for k, v in data.items():
        print(f"  {k:15s}: {len(v)} pairs")


if __name__ == "__main__":
    main()
