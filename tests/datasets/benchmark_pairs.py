"""Benchmark pair definitions for the SupeRANSAC evaluation.

Two tracks, both built from REAL inputs with REAL ground truth. Nothing here
filters correspondences with the ground truth, injects synthetic outliers, or
fabricates confidence scores.

Track A -- detector matches with calibrated ground truth (F and E, pose error):
    * PhotoTourism  (CVPR'20 RANSAC tutorial; GT R, t, K from the tutorial files)
    * ETH3D         (high-res multi-view; GT from the COLMAP calibration)
    * 7Scenes       (RGB-D indoor; GT from the dataset pose files)
    The exact image pairs are frozen in datasets/pairs/calibrated_pairs.json
    (generated once, deterministically, and committed) so every run uses the
    same pairs regardless of what else is on disk.

Track A -- detector matches with planar ground truth (H, corner error):
    * Homogr + EVD  (GT homography from the dataset)
    * HPatches      (if present; GT homography from the dataset)

Track B -- classic labelled tentative correspondences (high outlier ratios):
    * kusvod2 / AdelaideRMF / Multi-H  (F; original SIFT tentative matches with
      inlier labels; metric: epipolar error on the labelled inliers)
    * Homogr / EVD                     (H; original tentative matches; GT H)
    * AdelaideRMF / Multi-H planes     (H; labelled dominant plane)
"""

import glob
import json
import os

import cv2
import numpy as np

from datasets.features import (  # noqa: F401  (re-exported for the test scripts)
    ALL_DETECTORS,
    DEFAULT_DATA_DIR,
    EXCLUDED_F_SCENES,
    ROOT_DIR,
    SUPERANSAC_DIR,
    TESTS_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    load_cache,
)

PAIRS_DIR = os.path.join(TESTS_DIR, "datasets", "pairs")
CALIBRATED_PAIRS_FILE = os.path.join(PAIRS_DIR, "calibrated_pairs.json")
CALIBRATED_DEV_PAIRS_FILE = os.path.join(PAIRS_DIR, "calibrated_pairs_dev.json")

# Target composition of the frozen calibrated pair list (300 pairs).
CALIBRATED_TARGETS = {
    "PhotoTourism": {"palace_of_westminster": 100},
    "ETH3D": {"courtyard": 20, "electro": 15, "facade": 20, "kicker": 15, "playground": 10, "relief": 15, "terrace": 5},
    "7Scenes": {"chess": 40, "fire": 40, "heads": 20},
}
# Development split used for tuning only (disjoint from the test list above, seed 1, 150 pairs).
CALIBRATED_DEV_TARGETS = {
    "PhotoTourism": {"palace_of_westminster": 50},
    "ETH3D": {"courtyard": 10, "electro": 7, "facade": 10, "kicker": 8, "playground": 5, "relief": 7, "terrace": 3},
    "7Scenes": {"chess": 20, "fire": 20, "heads": 10},
}


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def _skew(t):
    t = np.asarray(t, dtype=np.float64).flatten()
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]], dtype=np.float64)


def make_ef_matrices(R, t, K1, K2):
    E = _skew(t) @ R
    E = E / (np.linalg.norm(E) + 1e-12)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    F = F / (np.linalg.norm(F) + 1e-12)
    return E, F


def sampson_errors(pts, F):
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def _image_hw(path):
    im = cv2.imread(path)
    if im is None:
        return None
    return im.shape[0], im.shape[1]


# --------------------------------------------------------------------------------------
# Track A: calibrated image pairs (F, E)
# --------------------------------------------------------------------------------------
def _calibrated_index(data_dir):
    """Build {dataset: {key: pair_dict}} for everything available on disk."""
    from datasets.phototourism import PhotoTourism
    from datasets.eth3d import ETH3D
    from datasets.seven_scenes import SevenScenes

    index = {"PhotoTourism": {}, "ETH3D": {}, "7Scenes": {}}

    pt_root = os.path.join(data_dir, "phototourism")
    if os.path.isdir(os.path.join(pt_root, "val")):
        ds = PhotoTourism(pt_root, split="val", max_pairs=0)
        for idx, (seq, key) in enumerate(zip(ds.seqs, ds.img_pairs)):
            index["PhotoTourism"][f"{seq}|{key}"] = ("pt", ds, idx)

    eth_root = os.path.join(data_dir, "eth3d")
    if os.path.isdir(eth_root):
        ds = ETH3D(eth_root, split="test", downsize_factor=4)
        for idx, d in enumerate(ds.data):
            scene = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(d["ref_img_path"]))))
            index["ETH3D"][f"{scene}|{d['ref_name']}|{d['target_name']}"] = ("eth", ds, idx)

    s7_root = os.path.join(data_dir, "7scenes")
    if os.path.isdir(s7_root):
        ds = SevenScenes(s7_root, split="test", scene="all")
        for idx, (p1, p2) in enumerate(ds.img_pairs):
            seq = os.path.basename(os.path.dirname(p1))
            scene = os.path.basename(os.path.dirname(os.path.dirname(p1)))
            f1 = os.path.basename(p1).split(".")[0]
            f2 = os.path.basename(p2).split(".")[0]
            index["7Scenes"][f"{scene}|{seq}|{f1}|{f2}"] = ("s7", ds, idx)
    return index


def _materialize_calibrated_pair(dataset, key, entry):
    kind, ds, idx = entry
    if kind == "pt":
        item = ds[idx]
        p1, p2 = item["path1"], item["path2"]
        R = np.asarray(item["R_1_2"], dtype=np.float64)
        t = np.asarray(item["T_1_2"], dtype=np.float64).flatten()
        K1 = np.asarray(item["K1"], dtype=np.float64)
        K2 = np.asarray(item["K2"], dtype=np.float64)
        depth1 = depth2 = None
        scene = key.replace("|", "_")
    elif kind == "eth":
        item = ds[idx]
        p1, p2 = item["path1"], item["path2"]
        R = np.asarray(item["R_1_2"], dtype=np.float64)
        t = np.asarray(item["T_1_2"], dtype=np.float64).flatten()
        K1 = np.asarray(item["K1"], dtype=np.float64)
        K2 = np.asarray(item["K2"], dtype=np.float64)
        depth1 = depth2 = None
        scene = key.replace("|", "_")
    else:
        p1, p2 = ds.img_pairs[idx]
        depth1, depth2 = ds.depth_pairs[idx]
        R = np.asarray(ds.R_1_2[idx], dtype=np.float64)
        t = np.asarray(ds.T_1_2[idx], dtype=np.float64).flatten()
        K1 = np.asarray(ds.K, dtype=np.float64)
        K2 = np.asarray(ds.K, dtype=np.float64)
        scene = key.replace("|", "_")
    hw1, hw2 = _image_hw(p1), _image_hw(p2)
    if hw1 is None or hw2 is None:
        return None
    E_gt, F_gt = make_ef_matrices(R, t, K1, K2)
    return {
        "dataset": dataset,
        "scene": scene,
        "cache_key": f"{dataset}|{key}",
        "im1": p1, "im2": p2,
        "depth1": depth1, "depth2": depth2,
        "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
        "K1": K1, "K2": K2, "R": R, "t": t,
        "E_gt": E_gt, "F_gt": F_gt,
        "H_gt": None, "gt_inliers": None,
    }


def _generate_calibrated_pair_list(index, targets, seed=0, exclude=()):
    """Deterministically pick a frozen pair list from what is available (optionally excluding keys)."""
    rng = np.random.default_rng(seed)
    excluded = {(e["dataset"], e["key"]) for e in exclude}
    chosen = []
    for dataset, per_scene in targets.items():
        keys = sorted(k for k in index[dataset].keys() if (dataset, k) not in excluded)
        for scene, n in per_scene.items():
            scene_keys = [k for k in keys if k.split("|")[0] == scene]
            if not scene_keys:
                print(f"[pairs] WARNING: no pairs available for {dataset}/{scene}")
                continue
            take = min(n, len(scene_keys))
            sel = rng.choice(len(scene_keys), size=take, replace=False)
            chosen.extend({"dataset": dataset, "key": scene_keys[i]} for i in sorted(sel))
    return chosen


def load_calibrated_pairs(data_dir=DEFAULT_DATA_DIR, split="test", verbose=True):
    """Return the frozen list of calibrated pairs (PhotoTourism, ETH3D, 7Scenes).

    split="test" -> tests/datasets/pairs/calibrated_pairs.json (300 pairs, used for all reported numbers)
    split="dev"  -> tests/datasets/pairs/calibrated_pairs_dev.json (150 disjoint pairs, used for tuning only)
    """
    index = _calibrated_index(data_dir)
    pairs_file = CALIBRATED_PAIRS_FILE if split == "test" else CALIBRATED_DEV_PAIRS_FILE
    if os.path.exists(pairs_file):
        with open(pairs_file, "r") as f:
            wanted = json.load(f)
    else:
        if split == "test":
            wanted = _generate_calibrated_pair_list(index, CALIBRATED_TARGETS, seed=0)
        else:
            with open(CALIBRATED_PAIRS_FILE, "r") as f:
                test_keys = json.load(f)
            wanted = _generate_calibrated_pair_list(index, CALIBRATED_DEV_TARGETS, seed=1, exclude=test_keys)
        os.makedirs(os.path.dirname(pairs_file), exist_ok=True)
        with open(pairs_file, "w") as f:
            json.dump(wanted, f, indent=1)
        print(f"[pairs] Wrote frozen {split} pair list with {len(wanted)} pairs to {pairs_file}")

    pairs, missing = [], {}
    for w in wanted:
        entry = index.get(w["dataset"], {}).get(w["key"])
        if entry is None:
            missing[w["dataset"]] = missing.get(w["dataset"], 0) + 1
            continue
        p = _materialize_calibrated_pair(w["dataset"], w["key"], entry)
        if p is None:
            missing[w["dataset"]] = missing.get(w["dataset"], 0) + 1
            continue
        if np.linalg.norm(p["t"]) < 1e-6:
            continue
        pairs.append(p)
    if verbose:
        counts = {}
        for p in pairs:
            counts[p["dataset"]] = counts.get(p["dataset"], 0) + 1
        print(f"[pairs] Calibrated {split} pairs loaded: {counts} (total {len(pairs)})")
        if missing:
            print(f"[pairs] WARNING: {missing} frozen pairs are missing on disk -- run tests/download_datasets.py")
    return pairs


# --------------------------------------------------------------------------------------
# Track A: planar image pairs (H) -- Homogr / EVD / HPatches with detector matches
# --------------------------------------------------------------------------------------
def _magsac_dir(data_dir, sub):
    d = os.path.join(data_dir, "magsac_data", sub)
    if not os.path.isdir(d):
        d = os.path.join(ROOT_DIR, "magsac_data", "data", sub)
    return d


EVD_SCENES = {
    "adam", "boat", "Boston", "BostonLib", "BruggeSquare", "BruggeTower", "Brussels", "CapitalRegion",
    "cf", "cat", "Eiffel", "ExtremeZoom", "graf", "piazza", "star", "table", "vin",
}


def _read_homogr_scene(base):
    """Read one homogr/EVD scene: images, GT H (A->B) and the labelled tentative correspondences."""
    pts_f = base + "_pts.txt"
    im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
    im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
    if not (os.path.exists(pts_f) and os.path.exists(im1_p) and os.path.exists(im2_p)):
        return None
    data = np.loadtxt(pts_f, dtype=np.float64)
    pts = data[:, [0, 1, 3, 4]].astype(np.float64)
    labels = data[:, 6]
    H_gt = np.linalg.inv(np.loadtxt(base + "_model.txt", dtype=np.float64))
    H_gt /= H_gt[2, 2] if abs(H_gt[2, 2]) > 1e-12 else 1.0
    gt_inl = pts[labels == 1]
    hw1, hw2 = _image_hw(im1_p), _image_hw(im2_p)
    if hw1 is None or hw2 is None or len(gt_inl) < 8:
        return None
    return {
        "im1": im1_p, "im2": im2_p,
        "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
        "H_gt": H_gt, "gt_inliers": gt_inl,
        "tentative": pts, "labels": labels,
    }


def load_planar_pairs(data_dir=DEFAULT_DATA_DIR, verbose=True):
    """Homogr + EVD (+ HPatches if present) with real GT homographies, for detector-match evaluation."""
    pairs = []
    magsac_h = _magsac_dir(data_dir, "homography")
    for pts_f in sorted(glob.glob(os.path.join(magsac_h, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        sc = _read_homogr_scene(base)
        if sc is None:
            continue
        ds_name = "EVD" if scene in EVD_SCENES else "Homogr"
        pairs.append({
            "dataset": ds_name, "scene": scene, "cache_key": f"h|{scene}",
            "im1": sc["im1"], "im2": sc["im2"],
            "h1": sc["h1"], "w1": sc["w1"], "h2": sc["h2"], "w2": sc["w2"],
            "K1": None, "K2": None, "R": None, "t": None, "E_gt": None, "F_gt": None,
            "H_gt": sc["H_gt"], "gt_inliers": sc["gt_inliers"],
        })

    # HPatches sequences (viewpoint subsets: v_*), GT homographies 1->k
    hp_dir = os.path.join(data_dir, "hpatches")
    if os.path.isdir(hp_dir):
        for seq_dir in sorted(glob.glob(os.path.join(hp_dir, "v_*"))):
            seq = os.path.basename(seq_dir)
            im1_p = os.path.join(seq_dir, "1.ppm")
            if not os.path.exists(im1_p):
                continue
            hw1 = _image_hw(im1_p)
            for k in range(2, 7):
                im2_p = os.path.join(seq_dir, f"{k}.ppm")
                h_p = os.path.join(seq_dir, f"H_1_{k}")
                if not (os.path.exists(im2_p) and os.path.exists(h_p)):
                    continue
                hw2 = _image_hw(im2_p)
                H_gt = np.loadtxt(h_p, dtype=np.float64)
                H_gt /= H_gt[2, 2]
                # Corner error is measured over the full image-1 frame for HPatches.
                corners = np.array([[0, 0], [hw1[1], 0], [hw1[1], hw1[0]], [0, hw1[0]]], dtype=np.float64)
                pairs.append({
                    "dataset": "HPatches", "scene": f"{seq}_1_{k}", "cache_key": f"hp|{seq}|1|{k}",
                    "im1": im1_p, "im2": im2_p,
                    "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
                    "K1": None, "K2": None, "R": None, "t": None, "E_gt": None, "F_gt": None,
                    "H_gt": H_gt, "gt_inliers": np.hstack([corners, corners]),
                })
    # Oxford affine-covariant sequences: img1..img6.ppm, H1to{k}p
    ox_dir = os.path.join(data_dir, "oxford_affine")
    if os.path.isdir(ox_dir):
        for seq_dir in sorted(glob.glob(os.path.join(ox_dir, "*"))):
            seq = os.path.basename(seq_dir)
            im1_p = os.path.join(seq_dir, "img1.ppm")
            if not os.path.exists(im1_p):
                continue
            hw1 = _image_hw(im1_p)
            for k in range(2, 7):
                im2_p = os.path.join(seq_dir, f"img{k}.ppm")
                h_p = os.path.join(seq_dir, f"H1to{k}p")
                if not (os.path.exists(im2_p) and os.path.exists(h_p)):
                    continue
                hw2 = _image_hw(im2_p)
                H_gt = np.loadtxt(h_p, dtype=np.float64)
                H_gt /= H_gt[2, 2]
                corners = np.array([[0, 0], [hw1[1], 0], [hw1[1], hw1[0]], [0, hw1[0]]], dtype=np.float64)
                pairs.append({
                    "dataset": "OxfordAffine", "scene": f"{seq}_1_{k}", "cache_key": f"ox|{seq}|1|{k}",
                    "im1": im1_p, "im2": im2_p,
                    "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
                    "K1": None, "K2": None, "R": None, "t": None, "E_gt": None, "F_gt": None,
                    "H_gt": H_gt, "gt_inliers": np.hstack([corners, corners]),
                })
    if verbose:
        counts = {}
        for p in pairs:
            counts[p["dataset"]] = counts.get(p["dataset"], 0) + 1
        print(f"[pairs] Planar pairs loaded: {counts} (total {len(pairs)})")
    return pairs


# --------------------------------------------------------------------------------------
# Track B: classic labelled tentative correspondences (no detector involved)
# --------------------------------------------------------------------------------------
def load_classic_f_problems(data_dir=DEFAULT_DATA_DIR, verbose=True):
    """kusvod2 / AdelaideRMF / Multi-H: original SIFT tentative matches with inlier labels."""
    magsac_f = _magsac_dir(data_dir, "fundamental_matrix")
    problems = []
    for pts_f in sorted(glob.glob(os.path.join(magsac_f, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        if scene in EXCLUDED_F_SCENES:
            continue
        im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
        im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
        hw1, hw2 = _image_hw(im1_p), _image_hw(im2_p)
        if hw1 is None or hw2 is None:
            continue
        data = np.loadtxt(pts_f, dtype=np.float64)
        pts = np.ascontiguousarray(data[:, [0, 1, 3, 4]], dtype=np.float64)
        labels = data[:, 6].astype(int)
        gt_inl = pts[labels > 0]
        if len(gt_inl) < 12:
            continue
        # Reference F from the labelled inliers (8-point LSQ on all labelled inliers).
        F_ref, _ = cv2.findFundamentalMat(gt_inl[:, :2], gt_inl[:, 2:4], cv2.FM_8POINT)
        if F_ref is None or F_ref.shape != (3, 3):
            continue
        ds_name = "kusvod2" if len(np.unique(labels)) <= 2 else "AdelaideRMF/Multi-H"
        problems.append({
            "dataset": ds_name, "scene": scene,
            "im1": im1_p, "im2": im2_p,
            "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
            "corrs": pts,                       # input: ALL tentative matches, dataset order
            "scores": np.ones(len(pts)),        # no confidences are provided by these datasets
            "labels": labels,
            "gt_inliers": np.ascontiguousarray(gt_inl),
            "F_ref": F_ref,
            "inlier_ratio": float(np.mean(labels > 0)),
        })
    if verbose:
        counts = {}
        for p in problems:
            counts[p["dataset"]] = counts.get(p["dataset"], 0) + 1
        ratios = [p["inlier_ratio"] for p in problems]
        print(f"[pairs] Classic F problems: {counts} (total {len(problems)}, "
              f"inlier ratio {np.min(ratios):.2f}-{np.max(ratios):.2f}, median {np.median(ratios):.2f})")
    return problems


def load_classic_h_problems(data_dir=DEFAULT_DATA_DIR, verbose=True):
    """Homogr / EVD original tentative matches + AdelaideRMF/Multi-H labelled dominant planes."""
    problems = []
    magsac_h = _magsac_dir(data_dir, "homography")
    for pts_f in sorted(glob.glob(os.path.join(magsac_h, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        sc = _read_homogr_scene(base)
        if sc is None:
            continue
        labels = sc["labels"]
        problems.append({
            "dataset": "EVD" if scene in EVD_SCENES else "Homogr", "scene": scene,
            "im1": sc["im1"], "im2": sc["im2"],
            "h1": sc["h1"], "w1": sc["w1"], "h2": sc["h2"], "w2": sc["w2"],
            "corrs": np.ascontiguousarray(sc["tentative"]),
            "scores": np.ones(len(sc["tentative"])),
            "labels": labels,
            "gt_inliers": sc["gt_inliers"], "H_gt": sc["H_gt"],
            "inlier_ratio": float(np.mean(labels == 1)),
        })

    # AdelaideRMF / Multi-H: all tentative matches as input, largest labelled plane as GT.
    magsac_f = _magsac_dir(data_dir, "fundamental_matrix")
    for pts_f in sorted(glob.glob(os.path.join(magsac_f, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        if scene in EXCLUDED_F_SCENES:
            continue
        data = np.loadtxt(pts_f, dtype=np.float64)
        pts = np.ascontiguousarray(data[:, [0, 1, 3, 4]], dtype=np.float64)
        labels = data[:, 6].astype(int)
        u_labels, counts = np.unique(labels[labels > 0], return_counts=True)
        if len(u_labels) < 2:
            continue  # single-structure (kusvod2) scenes are not multi-plane
        lbl = int(u_labels[np.argmax(counts)])
        plane_pts = pts[labels == lbl]
        if len(plane_pts) < 30:
            continue
        H_pl, inl_pl = cv2.findHomography(plane_pts[:, :2], plane_pts[:, 2:4], cv2.USAC_MAGSAC, 1.5)
        if H_pl is None or inl_pl is None or np.sum(inl_pl) < 20:
            continue
        clean = plane_pts[inl_pl.ravel() == 1]
        H_gt, _ = cv2.findHomography(clean[:, :2], clean[:, 2:4], 0)
        im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
        im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
        hw1, hw2 = _image_hw(im1_p), _image_hw(im2_p)
        if hw1 is None or hw2 is None:
            continue
        problems.append({
            "dataset": "AdelaideRMF/Multi-H", "scene": scene,
            "im1": im1_p, "im2": im2_p,
            "h1": hw1[0], "w1": hw1[1], "h2": hw2[0], "w2": hw2[1],
            "corrs": pts, "scores": np.ones(len(pts)), "labels": (labels == lbl).astype(int),
            "gt_inliers": np.ascontiguousarray(clean), "H_gt": H_gt,
            "inlier_ratio": float(np.mean(labels == lbl)),
        })
    if verbose:
        counts = {}
        for p in problems:
            counts[p["dataset"]] = counts.get(p["dataset"], 0) + 1
        ratios = [p["inlier_ratio"] for p in problems]
        print(f"[pairs] Classic H problems: {counts} (total {len(problems)}, "
              f"inlier ratio {np.min(ratios):.2f}-{np.max(ratios):.2f}, median {np.median(ratios):.2f})")
    return problems

