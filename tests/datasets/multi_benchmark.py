"""Unified multi-dataset benchmark loader and feature cache for SupeRANSAC tests.

Provides standardized access to:
1. Fundamental matrix (F) benchmark across 4 datasets:
   - PhotoTourism (Palace of Westminster)
   - ETH3D (courtyard, electro, office, relief)
   - 7Scenes (chess, fire, heads)
   - Kusvod2 & AdelaideRMF (classic epipolar benchmark)
2. Essential matrix (E) benchmark across 4 datasets:
   - PhotoTourism (Palace of Westminster)
   - ETH3D (courtyard, electro, office, relief)
   - 7Scenes (chess, fire, heads)
   - Strecha & Sacre Coeur (classic calibrated benchmark)
3. Homography (H) benchmark across 5 datasets:
   - Homogr (classic planar homography benchmark)
   - EVD (Extreme View Dataset)
   - PhotoTourism
   - ETH3D
   - 7Scenes
4. 6D Absolute Pose (PnP, 2D-3D) benchmark across 2 datasets:
   - 7Scenes RGB-D (real 2D-3D lifted from RGB-D depth maps)
   - pose6dscene (calibrated 2D-3D benchmark across 7 outlier regimes)
5. 6D Rigid Registration (SE(3), 3D-3D) benchmark across 2 datasets:
   - 7Scenes RGB-D (real 3D-3D point cloud pairs from RGB-D depth maps)
   - rigid_pose_example (3D-3D point cloud benchmark across 4 difficulty tiers)
"""

from concurrent.futures import ProcessPoolExecutor
import glob
import os
import pickle
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_HOME", "/usr/local/google/home/dbarath/ransac/.cache/torch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERANSAC_DIR = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(SUPERANSAC_DIR)
DEFAULT_DATA_DIR = os.path.join(SUPERANSAC_DIR, "data")

EXCLUDED_F_SCENES = {
    "Brussels", "Dresden", "Leuven1", "Leuven2", "dino1", "dino2", "temple1", "temple2", "neem"
}

DETECTOR_KEY_MAP = {
    "loma": "LoMa",
    "LoMa": "LoMa",
    "roma": "RoMa",
    "RoMa": "RoMa",
    "loftr": "LoFTR",
    "LoFTR": "LoFTR",
    "splg": "DISK+LG",
    "disk_lg": "DISK+LG",
    "DISK+LG": "DISK+LG",
    "rootsift": "RootSIFT",
    "RootSIFT": "RootSIFT",
    "sift": "SIFT",
    "SIFT": "SIFT",
    "orb": "ORB",
    "ORB": "ORB",
}

ALL_DETECTORS = ["LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"]


def canonical_detector_name(name):
    if name not in DETECTOR_KEY_MAP:
        raise ValueError(f"Unknown detector '{name}'. Supported: {list(DETECTOR_KEY_MAP.keys())}")
    return DETECTOR_KEY_MAP[name]


def get_cache_path(data_dir=DEFAULT_DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "multi_detector_cache.pkl")
    if not os.path.exists(cache_path):
        seed_candidates = [
            os.path.join(ROOT_DIR, "benchmark", "detector_matches_cache.pkl"),
            os.path.join(SUPERANSAC_DIR, "benchmarks", "detector_matches_cache.pkl"),
        ]
        for seed in seed_candidates:
            if os.path.exists(seed):
                shutil.copyfile(seed, cache_path)
                break
    return cache_path


def load_cache(data_dir=DEFAULT_DATA_DIR):
    cache_path = get_cache_path(data_dir)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache, data_dir=DEFAULT_DATA_DIR):
    cache_path = get_cache_path(data_dir)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def match_sift(im1_p, im2_p, root_sift=False, ratio_thresh=0.95):
    im1 = cv2.imread(im1_p, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2_p, cv2.IMREAD_GRAYSCALE)
    if im1 is None or im2 is None:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    sift = cv2.SIFT_create(4000)
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    if root_sift:
        des1 = np.sqrt(des1 / (np.linalg.norm(des1, ord=1, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        des2 = np.sqrt(des2 / (np.linalg.norm(des2, ord=1, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    corrs, scores = [], []
    for m_n in raw:
        if len(m_n) < 2:
            continue
        m, n = m_n
        ratio = m.distance / (n.distance + 1e-12)
        if ratio < ratio_thresh:
            corrs.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
            scores.append(max(1e-4, 1.0 - ratio))
    if not corrs:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    corrs = np.asarray(corrs, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    return np.ascontiguousarray(corrs[order]), np.ascontiguousarray(scores[order])


def match_orb(im1_p, im2_p, ratio_thresh=0.95):
    im1 = cv2.imread(im1_p, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2_p, cv2.IMREAD_GRAYSCALE)
    if im1 is None or im2 is None:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des1, des2, k=2)
    corrs, scores = [], []
    for m_n in raw:
        if len(m_n) < 2:
            continue
        m, n = m_n
        ratio = m.distance / (n.distance + 1e-12)
        if ratio < ratio_thresh:
            corrs.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
            scores.append(max(1e-4, 1.0 - ratio))
    if not corrs:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    corrs = np.asarray(corrs, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    return np.ascontiguousarray(corrs[order]), np.ascontiguousarray(scores[order])


_WORKER_MODELS = {}


def _extract_worker(task):
    key, im1_p, im2_p, needed_dets = task
    import torch

    torch.set_num_threads(1)
    device = torch.device("cpu")
    out = {}

    if "SIFT" in needed_dets:
        out["SIFT"] = match_sift(im1_p, im2_p, root_sift=False)
    if "RootSIFT" in needed_dets:
        out["RootSIFT"] = match_sift(im1_p, im2_p, root_sift=True)
    if "ORB" in needed_dets:
        out["ORB"] = match_orb(im1_p, im2_p)

    if "LoMa" in needed_dets:
        loma_src = os.path.join(ROOT_DIR, "LoMa", "src")
        if loma_src not in sys.path:
            sys.path.insert(0, loma_src)
        from loma.loma import LoMa, LoMaB, filter_matches, to_pixel_coords

        if "loma" not in _WORKER_MODELS:
            lm = LoMa(LoMaB(mp=False, compile=False, num_keypoints=1024))
            lm._detector.resize = 640
            _WORKER_MODELS["loma"] = lm
        lm = _WORKER_MODELS["loma"]
        with torch.inference_mode():
            kpA = lm._detector.detect_from_path(im1_p, num_keypoints=1024)["keypoints"]
            kpB = lm._detector.detect_from_path(im2_p, num_keypoints=1024)["keypoints"]
            descA = lm._descriptor.describe_keypoints_from_path(im1_p, kpA, H=448, W=448)["descriptions"]
            descB = lm._descriptor.describe_keypoints_from_path(im2_p, kpB, H=448, W=448)["descriptions"]
            w1_l, h1_l = Image.open(im1_p).size
            w2_l, h2_l = Image.open(im2_p).size
            scores = lm(kpA, kpB, descA, descB)["scores"]
            m0, _, mscores0, _ = filter_matches(scores, lm.cfg.filter_threshold)
            valid = m0[0] > -1
            if int(valid.sum()) == 0:
                out["LoMa"] = (np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64))
            else:
                matched_A = to_pixel_coords(kpA[0][torch.where(valid)[0]], h1_l, w1_l).cpu().numpy().astype(np.float64)
                matched_B = to_pixel_coords(kpB[0][m0[0][valid]], h2_l, w2_l).cpu().numpy().astype(np.float64)
                m_sc = mscores0[0][valid].cpu().numpy().astype(np.float64)
                c_loma = np.hstack([matched_A, matched_B])
                ord_lm = np.argsort(-m_sc)
                out["LoMa"] = (np.ascontiguousarray(c_loma[ord_lm]), np.ascontiguousarray(m_sc[ord_lm]))

    if any(d in needed_dets for d in ("RoMa", "LoFTR", "DISK+LG")):
        import kornia.feature as KF
        import romatch

        if "roma" not in _WORKER_MODELS and "RoMa" in needed_dets:
            _WORKER_MODELS["roma"] = romatch.roma_outdoor(
                device=device,
                coarse_res=280,
                upsample_preds=False,
                amp_dtype=torch.float32,
                use_custom_corr=False,
                symmetric=False,
            ).eval()
        if "loftr" not in _WORKER_MODELS and "LoFTR" in needed_dets:
            _WORKER_MODELS["loftr"] = KF.LoFTR(pretrained="outdoor").eval().to(device)
        if "disk" not in _WORKER_MODELS and "DISK+LG" in needed_dets:
            _WORKER_MODELS["disk"] = KF.DISK.from_pretrained("depth").eval().to(device)
            _WORKER_MODELS["lg"] = KF.LightGlue("disk").eval().to(device)

        if "RoMa" in needed_dets:
            with torch.inference_mode():
                r_model = _WORKER_MODELS["roma"]
                warp, certainty = r_model.match(im1_p, im2_p, device=device)
                matches, cert = r_model.sample(warp, certainty, num=1500)
                W_A, H_A = Image.open(im1_p).size
                W_B, H_B = Image.open(im2_p).size
                kptsA, kptsB = r_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                c_roma = np.hstack((kptsA.cpu().numpy(), kptsB.cpu().numpy())).astype(np.float64)
                s_roma = cert.cpu().numpy().astype(np.float64)
                ord_r = np.argsort(-s_roma)
                out["RoMa"] = (np.ascontiguousarray(c_roma[ord_r]), np.ascontiguousarray(s_roma[ord_r]))

        if "LoFTR" in needed_dets:
            im1_gray = cv2.imread(im1_p, cv2.IMREAD_GRAYSCALE)
            im2_gray = cv2.imread(im2_p, cv2.IMREAD_GRAYSCALE)
            h1, w1 = im1_gray.shape[:2]
            h2, w2 = im2_gray.shape[:2]
            tw, th = 640, 480
            im1_r = cv2.resize(im1_gray, (tw, th))
            im2_r = cv2.resize(im2_gray, (tw, th))
            with torch.inference_mode():
                inp = {
                    "image0": torch.from_numpy(im1_r).float()[None, None] / 255.0,
                    "image1": torch.from_numpy(im2_r).float()[None, None] / 255.0,
                }
                pred = _WORKER_MODELS["loftr"](inp)
                mk0 = pred["keypoints0"].cpu().numpy().astype(np.float64)
                mk1 = pred["keypoints1"].cpu().numpy().astype(np.float64)
                mconf = pred["confidence"].cpu().numpy().astype(np.float64)
                mk0[:, 0] *= w1 / float(tw)
                mk0[:, 1] *= h1 / float(th)
                mk1[:, 0] *= w2 / float(tw)
                mk1[:, 1] *= h2 / float(th)
                c_loftr = np.hstack([mk0, mk1])
                ord_l = np.argsort(-mconf)
                out["LoFTR"] = (np.ascontiguousarray(c_loftr[ord_l]), np.ascontiguousarray(mconf[ord_l]))

        if "DISK+LG" in needed_dets:
            im1_rgb = cv2.cvtColor(cv2.imread(im1_p), cv2.COLOR_BGR2RGB)
            im2_rgb = cv2.cvtColor(cv2.imread(im2_p), cv2.COLOR_BGR2RGB)
            h1, w1 = im1_rgb.shape[:2]
            h2, w2 = im2_rgb.shape[:2]
            dw1, dh1 = (min(640, (w1 // 16) * 16), min(480, (h1 // 16) * 16))
            dw2, dh2 = (min(640, (w2 // 16) * 16), min(480, (h2 // 16) * 16))
            t1 = torch.from_numpy(cv2.resize(im1_rgb, (dw1, dh1))).permute(2, 0, 1).float()[None] / 255.0
            t2 = torch.from_numpy(cv2.resize(im2_rgb, (dw2, dh2))).permute(2, 0, 1).float()[None] / 255.0
            with torch.inference_mode():
                f1 = _WORKER_MODELS["disk"](t1, n=1024, pad_if_not_divisible=True)[0]
                f2 = _WORKER_MODELS["disk"](t2, n=1024, pad_if_not_divisible=True)[0]
                kp1_d = f1.keypoints
                kp2_d = f2.keypoints
                lg_in = {
                    "image0": {
                        "keypoints": kp1_d[None],
                        "descriptors": f1.descriptors[None],
                        "image_size": torch.tensor([[dw1, dh1]], dtype=torch.float32),
                    },
                    "image1": {
                        "keypoints": kp2_d[None],
                        "descriptors": f2.descriptors[None],
                        "image_size": torch.tensor([[dw2, dh2]], dtype=torch.float32),
                    },
                }
                lg_out = _WORKER_MODELS["lg"](lg_in)
                m_idx = lg_out["matches"][0].cpu().numpy()
                m_sc = lg_out["scores"][0].cpu().numpy().astype(np.float64)
                if len(m_idx) > 0:
                    p0 = kp1_d.cpu().numpy()[m_idx[:, 0]].astype(np.float64)
                    p1 = kp2_d.cpu().numpy()[m_idx[:, 1]].astype(np.float64)
                    p0[:, 0] *= w1 / float(dw1)
                    p0[:, 1] *= h1 / float(dh1)
                    p1[:, 0] *= w2 / float(dw2)
                    p1[:, 1] *= h2 / float(dh2)
                    c_dlg = np.hstack([p0, p1])
                    ord_d = np.argsort(-m_sc)
                    out["DISK+LG"] = (np.ascontiguousarray(c_dlg[ord_d]), np.ascontiguousarray(m_sc[ord_d]))
                else:
                    out["DISK+LG"] = (np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64))

    return key, out


def expand_challenging_regimes(cache, pairs, detectors):
    """Populate challenging cluttered / multi-structure / noisy-prior regimes from cached detector matches."""
    for det in detectors:
        rng_f = np.random.default_rng(1001)
        rng_h = np.random.default_rng(3003)
        for p in pairs:
            key = p["cache_key"]
            base_key = p.get("base_cache_key", key)
            regime = p.get("regime", "raw")
            if regime == "raw" or base_key not in cache or det not in cache[base_key]:
                continue
            corrs, scores = cache[base_key][det]
            h1, w1, h2, w2 = float(p["h1"]), float(p["w1"]), float(p["h2"]), float(p["w2"])

            if regime in ("ord1", "ord2"):
                F_gt = p["F_gt"]
                r_det = _sampson_errors(corrs, F_gt) if len(corrs) >= 8 else np.array([])
                det_inl = corrs[r_det < 2.2] if len(corrs) >= 8 else np.empty((0, 4))
                det_out = corrs[r_det >= 3.5] if len(corrs) >= 8 else np.empty((0, 4))
                target_inl = min(max(36, len(det_inl) // 2), 110)
                if len(det_inl) > target_inl:
                    idx_sub = np.linspace(0, len(det_inl) - 1, target_inl, dtype=int)
                    use_inl = det_inl[idx_sub]
                else:
                    use_inl = det_inl

                if regime == "ord1":
                    n_bg = max(140, int(1.6 * max(len(use_inl), 20)))
                    extra = np.column_stack([
                        rng_f.uniform(0, w1, n_bg),
                        rng_f.uniform(0, h1, n_bg),
                        rng_f.uniform(0, w2, n_bg),
                        rng_f.uniform(0, h2, n_bg),
                    ])
                    parts_f = [x for x in (use_inl, det_out[:80] if len(det_out) > 0 else extra[:40], extra) if len(x) > 0]
                    c_out = np.vstack(parts_f)
                    sc_out = np.zeros(len(c_out), dtype=np.float64)
                    sc_out[:len(use_inl)] = rng_f.beta(3.2, 1.4, size=len(use_inl))
                    sc_out[len(use_inl):] = rng_f.beta(1.4, 2.6, size=len(c_out) - len(use_inl))
                    ord_i = np.argsort(-sc_out)
                    c_out, sc_out = c_out[ord_i], sc_out[ord_i]
                else:
                    use_inl2 = use_inl[:min(len(use_inl), 85)]
                    n_extra2 = max(180, int(2.5 * max(len(use_inl2), 20)))
                    extra2 = np.column_stack([
                        rng_f.uniform(0, w1, n_extra2),
                        rng_f.uniform(0, h1, n_extra2),
                        rng_f.uniform(0, w2, n_extra2),
                        rng_f.uniform(0, h2, n_extra2),
                    ])
                    c_out = np.vstack([use_inl2, extra2]) if len(use_inl2) > 0 else extra2
                    c_out = c_out[rng_f.permutation(len(c_out))]
                    sc_out = np.linspace(1.0, 0.01, len(c_out))

                cache.setdefault(key, {})[det] = (
                    np.ascontiguousarray(c_out, dtype=np.float64),
                    np.ascontiguousarray(sc_out, dtype=np.float64),
                )

            elif regime in ("clut", "plane"):
                H_gt = p["H_gt"]
                if len(corrs) >= 4 and H_gt is not None:
                    p1_h = np.hstack([corrs[:, :2], np.ones((len(corrs), 1))])
                    p2_h = p1_h @ H_gt.T
                    p2_h = p2_h[:, :2] / (p2_h[:, 2:3] + 1e-12)
                    reproj = np.linalg.norm(p2_h - corrs[:, 2:4], axis=1)
                    det_inl = corrs[reproj < 3.5]
                    det_off = corrs[reproj > 12.0]
                else:
                    det_inl = np.empty((0, 4))
                    det_off = np.empty((0, 4))

                if regime == "clut":
                    target_h = min(max(18, len(det_inl) // 8), 32)
                    if len(det_inl) >= target_h:
                        sub_inl = det_inl[rng_h.choice(len(det_inl), target_h, replace=False)]
                    else:
                        sub_inl = det_inl
                    n_extra = max(140, max(len(sub_inl), 12) * 9)
                    extra = np.column_stack([
                        rng_h.uniform(0, w1, n_extra),
                        rng_h.uniform(0, h1, n_extra),
                        rng_h.uniform(0, w2, n_extra),
                        rng_h.uniform(0, h2, n_extra),
                    ])
                    c_out = np.vstack([sub_inl, extra]) if len(sub_inl) > 0 else extra
                    c_out = c_out[rng_h.permutation(len(c_out))]
                    sc_out = np.linspace(1.0, 0.01, len(c_out))
                else:
                    target_pl = min(max(20, len(det_inl) // 6), 36)
                    if len(det_inl) >= target_pl:
                        sub_inl = det_inl[rng_h.choice(len(det_inl), target_pl, replace=False)]
                    else:
                        sub_inl = det_inl
                    sub_off = det_off[:70] if len(det_off) > 0 else np.empty((0, 4))
                    n_extra = int(rng_h.integers(140, 260))
                    extra = np.column_stack([
                        rng_h.uniform(0, w1, n_extra),
                        rng_h.uniform(0, h1, n_extra),
                        rng_h.uniform(0, w2, n_extra),
                        rng_h.uniform(0, h2, n_extra),
                    ])
                    parts_h = [x for x in (sub_inl, sub_off, extra) if len(x) > 0]
                    c_out = np.vstack(parts_h)
                    c_out = c_out[rng_h.permutation(len(c_out))]
                    sc_out = np.linspace(1.0, 0.01, len(c_out))

                cache.setdefault(key, {})[det] = (
                    np.ascontiguousarray(c_out, dtype=np.float64),
                    np.ascontiguousarray(sc_out, dtype=np.float64),
                )
    return cache


def ensure_cached_matches(pairs, detectors, data_dir=DEFAULT_DATA_DIR, max_workers=12):
    """Ensure all (pair['cache_key'], detector) entries exist in cache and return cache."""
    detectors = [canonical_detector_name(d) for d in detectors]
    cache = load_cache(data_dir)
    tasks = {}
    for p in pairs:
        base_k = p.get("base_cache_key", p["cache_key"])
        existing = cache.get(base_k, {})
        needed = [d for d in detectors if d not in existing]
        if needed and base_k not in tasks:
            tasks[base_k] = (base_k, p["im1"], p["im2"], needed)

    if tasks:
        task_list = list(tasks.values())
        n_workers = min(max_workers, len(task_list))
        print(f"Extracting {detectors} features for {len(task_list)} image pairs using {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, (k, out) in enumerate(ex.map(_extract_worker, task_list), 1):
                if k not in cache:
                    cache[k] = {}
                cache[k].update(out)
                if idx % 10 == 0 or idx == len(task_list):
                    print(f"  Extracted {idx}/{len(task_list)} pairs...")
                    save_cache(cache, data_dir)
    return expand_challenging_regimes(cache, pairs, detectors)


def _sampson_errors(pts, F):
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def _skew(t):
    t = np.asarray(t, dtype=np.float64).flatten()
    return np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ], dtype=np.float64)


def _make_ef_matrices(R, t, K1, K2):
    E = _skew(t) @ R
    E = E / (np.linalg.norm(E) + 1e-12)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    F = F / (np.linalg.norm(F) + 1e-12)
    return E, F


def _append_fe_regimes(pairs, base_entry, add_ord1=True, add_ord2=False):
    pairs.append(base_entry)
    if add_ord1:
        p1 = dict(base_entry)
        p1["scene"] = f"{base_entry['scene']}_ord1"
        p1["cache_key"] = f"{base_entry['cache_key']}_ord1"
        p1["base_cache_key"] = base_entry["cache_key"]
        p1["regime"] = "ord1"
        pairs.append(p1)
    if add_ord2:
        p2 = dict(base_entry)
        p2["scene"] = f"{base_entry['scene']}_ord2"
        p2["cache_key"] = f"{base_entry['cache_key']}_ord2"
        p2["base_cache_key"] = base_entry["cache_key"]
        p2["regime"] = "ord2"
        pairs.append(p2)


def load_phototourism_pairs(data_dir=DEFAULT_DATA_DIR, max_pairs=15, include_regimes=True):
    from datasets.phototourism import PhotoTourism

    pt_root = os.path.join(data_dir, "phototourism")
    if not os.path.isdir(os.path.join(pt_root, "val")):
        return []
    ds = PhotoTourism(pt_root, split="val", max_pairs=max_pairs)
    pairs = []
    for idx in range(min(max_pairs, len(ds))):
        item = ds[idx]
        p1, p2 = item["path1"], item["path2"]
        im1 = cv2.imread(p1)
        im2 = cv2.imread(p2)
        if im1 is None or im2 is None:
            continue
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        R = np.asarray(item["R_1_2"], dtype=np.float64)
        t = np.asarray(item["T_1_2"], dtype=np.float64).flatten()
        K1 = np.asarray(item["K1"], dtype=np.float64)
        K2 = np.asarray(item["K2"], dtype=np.float64)
        E_gt, F_gt = _make_ef_matrices(R, t, K1, K2)
        m = ds.img_pairs[idx]
        base = {
            "dataset": "PhotoTourism",
            "scene": f"westminster_{m}",
            "cache_key": f"pt_{m}",
            "base_cache_key": f"pt_{m}",
            "regime": "raw",
            "im1": p1,
            "im2": p2,
            "h1": h1, "w1": w1, "h2": h2, "w2": w2,
            "K1": K1, "K2": K2,
            "R": R, "t": t,
            "E_gt": E_gt, "F_gt": F_gt,
            "gt_inliers": None,
            "H_gt": None,
        }
        _append_fe_regimes(pairs, base, add_ord1=include_regimes, add_ord2=(include_regimes and idx % 2 == 0))
    return pairs


ETH3D_BENCHMARK_SCENES = ("courtyard", "electro", "facade", "kicker", "office", "playground", "relief")
SEVEN_SCENES_BENCHMARK = ("chess", "fire", "heads")


def load_eth3d_pairs(data_dir=DEFAULT_DATA_DIR, max_pairs_per_scene=5, include_regimes=True):
    from datasets.eth3d import ETH3D

    eth_root = os.path.join(data_dir, "eth3d")
    if not os.path.isdir(eth_root):
        return []
    ds = ETH3D(eth_root, split="test", downsize_factor=4)
    by_scene = {}
    for idx, d in enumerate(ds.data):
        p1 = d["ref_img_path"]
        scene = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p1))))
        if scene in ETH3D_BENCHMARK_SCENES:
            by_scene.setdefault(scene, []).append(idx)

    pairs = []
    for scene in sorted(by_scene.keys()):
        indices = by_scene[scene]
        step = max(1, len(indices) // max_pairs_per_scene)
        chosen = indices[::step][:max_pairs_per_scene]
        for j, idx in enumerate(chosen):
            item = ds[idx]
            p1, p2 = item["path1"], item["path2"]
            im1 = cv2.imread(p1)
            im2 = cv2.imread(p2)
            if im1 is None or im2 is None:
                continue
            h1, w1 = im1.shape[:2]
            h2, w2 = im2.shape[:2]
            R = np.asarray(item["R_1_2"], dtype=np.float64)
            t = np.asarray(item["T_1_2"], dtype=np.float64).flatten()
            K1 = np.asarray(item["K1"], dtype=np.float64)
            K2 = np.asarray(item["K2"], dtype=np.float64)
            E_gt, F_gt = _make_ef_matrices(R, t, K1, K2)
            b1 = os.path.splitext(os.path.basename(p1))[0]
            b2 = os.path.splitext(os.path.basename(p2))[0]
            base = {
                "dataset": "ETH3D",
                "scene": f"{scene}_{b1}_{b2}",
                "cache_key": f"eth3d_{scene}_{b1}_{b2}",
                "base_cache_key": f"eth3d_{scene}_{b1}_{b2}",
                "regime": "raw",
                "im1": p1,
                "im2": p2,
                "h1": h1, "w1": w1, "h2": h2, "w2": w2,
                "K1": K1, "K2": K2,
                "R": R, "t": t,
                "E_gt": E_gt, "F_gt": F_gt,
                "gt_inliers": None,
                "H_gt": None,
            }
            _append_fe_regimes(pairs, base, add_ord1=include_regimes, add_ord2=(include_regimes and j % 2 == 0))
    return pairs


def load_7scenes_pairs(data_dir=DEFAULT_DATA_DIR, max_pairs_per_scene=6, include_regimes=True):
    from datasets.seven_scenes import SevenScenes

    s7_root = os.path.join(data_dir, "7scenes")
    if not os.path.isdir(s7_root):
        return []
    ds = SevenScenes(s7_root, split="test", scene="all")
    by_scene = {}
    for idx, (p1, p2) in enumerate(ds.img_pairs):
        scene = os.path.basename(os.path.dirname(os.path.dirname(p1)))
        if scene in SEVEN_SCENES_BENCHMARK:
            by_scene.setdefault(scene, []).append(idx)

    pairs = []
    for scene in sorted(by_scene.keys()):
        indices = by_scene[scene]
        step = max(1, len(indices) // max_pairs_per_scene)
        chosen = indices[::step][:max_pairs_per_scene]
        for j, idx in enumerate(chosen):
            p1, p2 = ds.img_pairs[idx]
            d1, d2 = ds.depth_pairs[idx]
            im1 = cv2.imread(p1)
            im2 = cv2.imread(p2)
            if im1 is None or im2 is None:
                continue
            h1, w1 = im1.shape[:2]
            h2, w2 = im2.shape[:2]
            R = np.asarray(ds.R_1_2[idx], dtype=np.float64)
            t = np.asarray(ds.T_1_2[idx], dtype=np.float64).flatten()
            if np.linalg.norm(t) < 0.05:
                continue
            K1 = np.asarray(ds.K, dtype=np.float64)
            K2 = np.asarray(ds.K, dtype=np.float64)
            E_gt, F_gt = _make_ef_matrices(R, t, K1, K2)
            seq = os.path.basename(os.path.dirname(p1))
            f1 = os.path.basename(p1).split(".")[0]
            f2 = os.path.basename(p2).split(".")[0]
            base = {
                "dataset": "7Scenes",
                "scene": f"{scene}_{seq}_{f1}_{f2}",
                "cache_key": f"7s_{scene}_{seq}_{f1}_{f2}",
                "base_cache_key": f"7s_{scene}_{seq}_{f1}_{f2}",
                "regime": "raw",
                "im1": p1,
                "im2": p2,
                "depth1": d1,
                "depth2": d2,
                "h1": h1, "w1": w1, "h2": h2, "w2": w2,
                "K1": K1, "K2": K2,
                "R": R, "t": t,
                "E_gt": E_gt, "F_gt": F_gt,
                "gt_inliers": None,
                "H_gt": None,
            }
            _append_fe_regimes(pairs, base, add_ord1=include_regimes, add_ord2=(include_regimes and j % 2 == 0))
    return pairs


def load_classic_f_pairs(data_dir=DEFAULT_DATA_DIR, include_regimes=True):
    magsac_dir = os.path.join(data_dir, "magsac_data", "fundamental_matrix")
    if not os.path.isdir(magsac_dir):
        magsac_dir = os.path.join(ROOT_DIR, "magsac_data", "data", "fundamental_matrix")
    pairs = []
    for pts_f in sorted(glob.glob(os.path.join(magsac_dir, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        if scene in EXCLUDED_F_SCENES:
            continue
        im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
        im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
        if not os.path.exists(im1_p) or not os.path.exists(im2_p):
            continue
        data = np.loadtxt(pts_f, dtype=np.float64)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6].astype(int)
        gt_inl = pts[labels > 0]
        if len(gt_inl) < 12:
            continue
        F_init, _ = cv2.findFundamentalMat(gt_inl[:, :2], gt_inl[:, 2:4], cv2.USAC_MAGSAC, 2.0, 0.99999, 10000)
        if F_init is None or F_init.shape != (3, 3):
            F_init, _ = cv2.findFundamentalMat(gt_inl[:, :2], gt_inl[:, 2:4], cv2.FM_8POINT)
        consistent = pts[_sampson_errors(pts, F_init) < 2.0]
        if len(consistent) >= 12:
            gt_inl = consistent
        F_gt, _ = cv2.findFundamentalMat(gt_inl[:, :2], gt_inl[:, 2:4], cv2.FM_8POINT)
        if F_gt is None or F_gt.shape != (3, 3):
            continue
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        f1 = max(w1, h1) * 1.1
        f2 = max(w2, h2) * 1.1
        K1 = np.array([[f1, 0, w1 / 2.0], [0, f1, h1 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        K2 = np.array([[f2, 0, w2 / 2.0], [0, f2, h2 / 2.0], [0, 0, 1.0]], dtype=np.float64)
        E_gt = K2.T @ F_gt @ K1
        p1_n = cv2.undistortPoints(gt_inl[:, :2].reshape(-1, 1, 2), K1, None)
        p2_n = cv2.undistortPoints(gt_inl[:, 2:4].reshape(-1, 1, 2), K2, None)
        _, R_gt, t_gt, _ = cv2.recoverPose(E_gt, p1_n, p2_n)
        t_gt = t_gt.flatten()
        base_entry = {
            "dataset": "Kusvod2/AdelaideRMF",
            "scene": scene,
            "cache_key": f"f_{scene}",
            "base_cache_key": f"f_{scene}",
            "regime": "raw",
            "im1": im1_p,
            "im2": im2_p,
            "h1": h1, "w1": w1, "h2": h2, "w2": w2,
            "K1": K1, "K2": K2,
            "R": R_gt, "t": t_gt,
            "E_gt": E_gt, "F_gt": F_gt,
            "gt_inliers": gt_inl,
            "H_gt": None,
        }
        _append_fe_regimes(pairs, base_entry, add_ord1=include_regimes, add_ord2=include_regimes)
    return pairs


def load_classic_e_pairs(data_dir=DEFAULT_DATA_DIR):
    magsac_e = os.path.join(data_dir, "magsac_data", "essential_matrix")
    if not os.path.isdir(magsac_e):
        magsac_e = os.path.join(ROOT_DIR, "magsac_data", "data", "essential_matrix")
    examples_dir = os.path.join(SUPERANSAC_DIR, "examples")
    pairs = []
    im1_p = os.path.join(magsac_e, "fountain1.jpg")
    im2_p = os.path.join(magsac_e, "fountain2.jpg")
    if os.path.exists(im1_p) and os.path.exists(im2_p):
        K1 = np.loadtxt(os.path.join(magsac_e, "fountain1.K"), dtype=np.float64)
        K2 = np.loadtxt(os.path.join(magsac_e, "fountain2.K"), dtype=np.float64)
        f_pts = np.loadtxt(os.path.join(magsac_e, "fountain_pts.txt"), skiprows=1, dtype=np.float64)
        p1_n = cv2.undistortPoints(f_pts[:, :2].reshape(-1, 1, 2), K1, None)
        p2_n = cv2.undistortPoints(f_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
        E_f_gt, inl_f = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
        _, R_gt, t_gt, _ = cv2.recoverPose(E_f_gt, p1_n[inl_f.ravel() == 1], p2_n[inl_f.ravel() == 1])
        t_gt = t_gt.flatten()
        E_gt, F_gt = _make_ef_matrices(R_gt, t_gt, K1, K2)
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        pairs.append({
            "dataset": "Strecha/SacreCoeur",
            "scene": "fountain",
            "cache_key": "e_fountain",
            "im1": im1_p, "im2": im2_p,
            "h1": im1.shape[0], "w1": im1.shape[1],
            "h2": im2.shape[0], "w2": im2.shape[1],
            "K1": K1, "K2": K2, "R": R_gt, "t": t_gt,
            "E_gt": E_gt, "F_gt": F_gt,
            "gt_inliers": None, "H_gt": None,
        })

    sc_im1 = os.path.join(examples_dir, "02085496_6952371977.jpg")
    sc_im2 = os.path.join(examples_dir, "02928139_3448003521.jpg")
    if os.path.exists(sc_im1) and os.path.exists(sc_im2):
        K1 = np.array([[1.23880078e03, 0.0, 3.90000000e02], [0.0, 1.23880078e03, 5.31500000e02], [0.0, 0.0, 1.0]], dtype=np.float64)
        K2 = np.array([[1.15996667e03, 0.0, 5.15000000e02], [0.0, 1.15996667e03, 3.43500000e02], [0.0, 0.0, 1.0]], dtype=np.float64)
        c_sc, _ = match_sift(sc_im1, sc_im2, ratio_thresh=0.72)
        p1_sc = cv2.undistortPoints(c_sc[:, :2].reshape(-1, 1, 2), K1, None)
        p2_sc = cv2.undistortPoints(c_sc[:, 2:4].reshape(-1, 1, 2), K2, None)
        E_sc_gt, inl_sc = cv2.findEssentialMat(p1_sc, p2_sc, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
        _, R_gt, t_gt, _ = cv2.recoverPose(E_sc_gt, p1_sc[inl_sc.ravel() == 1], p2_sc[inl_sc.ravel() == 1])
        t_gt = t_gt.flatten()
        E_gt, F_gt = _make_ef_matrices(R_gt, t_gt, K1, K2)
        im1 = cv2.imread(sc_im1)
        im2 = cv2.imread(sc_im2)
        pairs.append({
            "dataset": "Strecha/SacreCoeur",
            "scene": "sacre_coeur",
            "cache_key": "e_sacre_coeur",
            "im1": sc_im1, "im2": sc_im2,
            "h1": im1.shape[0], "w1": im1.shape[1],
            "h2": im2.shape[0], "w2": im2.shape[1],
            "K1": K1, "K2": K2, "R": R_gt, "t": t_gt,
            "E_gt": E_gt, "F_gt": F_gt,
            "gt_inliers": None, "H_gt": None,
        })
    return pairs


def load_classic_h_pairs(data_dir=DEFAULT_DATA_DIR, include_regimes=True):
    magsac_h = os.path.join(data_dir, "magsac_data", "homography")
    if not os.path.isdir(magsac_h):
        magsac_h = os.path.join(ROOT_DIR, "magsac_data", "data", "homography")
    magsac_f = os.path.join(data_dir, "magsac_data", "fundamental_matrix")
    if not os.path.isdir(magsac_f):
        magsac_f = os.path.join(ROOT_DIR, "magsac_data", "data", "fundamental_matrix")
    evd_scenes = {
        "adam", "boat", "Boston", "BostonLib", "BruggeSquare", "BruggeTower",
        "Brussels", "CapitalRegion", "cf", "cat", "Eiffel", "ExtremeZoom",
        "graf", "piazza", "star", "table", "vin",
    }
    pairs = []
    for pts_f in sorted(glob.glob(os.path.join(magsac_h, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
        im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
        if not os.path.exists(im1_p) or not os.path.exists(im2_p):
            continue
        data = np.loadtxt(pts_f, dtype=np.float64)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6]
        H_gt = np.linalg.inv(np.loadtxt(base + "_model.txt", dtype=np.float64))
        H_gt /= H_gt[2, 2] if abs(H_gt[2, 2]) > 1e-12 else 1.0
        gt_inl = pts[labels == 1]
        tent = pts[labels == 0]
        if len(tent) > 0:
            p1_h = np.hstack([tent[:, :2], np.ones((len(tent), 1))])
            p2_h = p1_h @ H_gt.T
            p2_h = p2_h[:, :2] / (p2_h[:, 2:3] + 1e-12)
            tent_inl = tent[np.linalg.norm(p2_h - tent[:, 2:4], axis=1) < 3.5]
            if len(tent_inl) >= 8:
                all_gt_pts = np.vstack([gt_inl, tent_inl])
                H_gt, _ = cv2.findHomography(all_gt_pts[:, :2], all_gt_pts[:, 2:4], 0)
                gt_inl = all_gt_pts
        im1 = cv2.imread(im1_p)
        im2 = cv2.imread(im2_p)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        ds_name = "EVD" if scene in evd_scenes else "Homogr"
        raw_entry = {
            "dataset": ds_name,
            "scene": scene,
            "cache_key": f"h_{scene}",
            "base_cache_key": f"h_{scene}",
            "regime": "raw",
            "im1": im1_p,
            "im2": im2_p,
            "h1": h1, "w1": w1, "h2": h2, "w2": w2,
            "K1": None, "K2": None,
            "R": None, "t": None,
            "E_gt": None, "F_gt": None,
            "gt_inliers": gt_inl,
            "H_gt": H_gt,
        }
        pairs.append(raw_entry)
        if include_regimes:
            clut_entry = dict(raw_entry)
            clut_entry["scene"] = f"{scene}_clut"
            clut_entry["cache_key"] = f"h_{scene}_clut"
            clut_entry["regime"] = "clut"
            pairs.append(clut_entry)

    # Dominant labeled planes from AdelaideRMF / Multi-H
    for pts_f in sorted(glob.glob(os.path.join(magsac_f, "*_pts.txt"))):
        base = pts_f[:-8]
        scene = os.path.basename(base)
        if scene in EXCLUDED_F_SCENES:
            continue
        im1_p = base + "A.png" if os.path.exists(base + "A.png") else base + "A.jpg"
        im2_p = base + "B.png" if os.path.exists(base + "B.png") else base + "B.jpg"
        if not os.path.exists(im1_p) or not os.path.exists(im2_p):
            continue
        data = np.loadtxt(pts_f, dtype=np.float64)
        pts = data[:, [0, 1, 3, 4]].astype(np.float64)
        labels = data[:, 6].astype(int)
        u_labels, counts = np.unique(labels, return_counts=True)
        for lbl, cnt in zip(u_labels, counts):
            if lbl > 0 and cnt >= 30:
                plane_pts = pts[labels == lbl]
                H_pl, inl_pl = cv2.findHomography(plane_pts[:, :2], plane_pts[:, 2:4], cv2.USAC_MAGSAC, 1.5)
                if H_pl is not None and inl_pl is not None and np.sum(inl_pl) >= 20:
                    clean_plane = plane_pts[inl_pl.ravel() == 1]
                    H_pl, _ = cv2.findHomography(clean_plane[:, :2], clean_plane[:, 2:4], 0)
                    im1 = cv2.imread(im1_p)
                    im2 = cv2.imread(im2_p)
                    h1, w1 = im1.shape[:2]
                    h2, w2 = im2.shape[:2]
                    pairs.append({
                        "dataset": "AdelaideRMF/Multi-H",
                        "scene": f"plane_{scene}",
                        "cache_key": f"h_plane_{scene}",
                        "base_cache_key": f"f_{scene}",
                        "regime": "plane",
                        "im1": im1_p,
                        "im2": im2_p,
                        "h1": h1, "w1": w1, "h2": h2, "w2": w2,
                        "K1": None, "K2": None,
                        "R": None, "t": None,
                        "E_gt": None, "F_gt": None,
                        "gt_inliers": clean_plane,
                        "H_gt": H_pl,
                    })
                    break
    return pairs


def get_fundamental_benchmark_pairs(data_dir=DEFAULT_DATA_DIR):
    """Returns all Fundamental matrix benchmark pairs across 4 datasets."""
    pairs = []
    pairs.extend(load_phototourism_pairs(data_dir))
    pairs.extend(load_eth3d_pairs(data_dir))
    pairs.extend(load_7scenes_pairs(data_dir))
    pairs.extend(load_classic_f_pairs(data_dir))
    return pairs


def get_essential_benchmark_pairs(data_dir=DEFAULT_DATA_DIR):
    """Returns all Essential matrix benchmark pairs across 5 datasets."""
    pairs = []
    for p in (
        load_phototourism_pairs(data_dir, include_regimes=True)
        + load_eth3d_pairs(data_dir, include_regimes=True)
        + load_7scenes_pairs(data_dir, include_regimes=True)
        + load_classic_e_pairs(data_dir)
        + load_classic_f_pairs(data_dir, include_regimes=True)
    ):
        if p.get("regime", "raw") != "ord2":
            pairs.append(p)
    return pairs


def get_homography_benchmark_pairs(data_dir=DEFAULT_DATA_DIR):
    """Returns all Homography benchmark pairs across EVD, Homogr, AdelaideRMF/Multi-H, ETH3D, and PhotoTourism."""
    pairs = []
    pairs.extend(load_classic_h_pairs(data_dir))
    pairs.extend(load_eth3d_pairs(data_dir, include_regimes=False))
    pairs.extend(load_phototourism_pairs(data_dir, include_regimes=False))
    return pairs


def _calibrate_7scenes_rgbd_pair(p, cache, detectors):
    """Calibrate exact 3D-3D and 2D-3D ground truth on Kinect depth maps for a 7Scenes pair."""
    d1_path, d2_path = p.get("depth1"), p.get("depth2")
    if not d1_path or not d2_path or not os.path.exists(d1_path) or not os.path.exists(d2_path):
        return None
    d1 = cv2.imread(d1_path, cv2.IMREAD_UNCHANGED)
    d2 = cv2.imread(d2_path, cv2.IMREAD_UNCHANGED)
    if d1 is None or d2 is None:
        return None
    K_d = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_d_inv = np.linalg.inv(K_d)
    K_rgb = p["K2"].astype(np.float64)

    pooled_c, pooled_X1, pooled_X2 = [], [], []
    for det in detectors:
        det_key = canonical_detector_name(det)
        corrs, _ = cache[p["cache_key"]][det_key]
        if len(corrs) < 12:
            continue
        u1 = np.round(corrs[:, 0]).astype(int)
        v1 = np.round(corrs[:, 1]).astype(int)
        u2 = np.round(corrs[:, 2]).astype(int)
        v2 = np.round(corrs[:, 3]).astype(int)
        valid = (
            (u1 >= 0) & (u1 < d1.shape[1]) & (v1 >= 0) & (v1 < d1.shape[0]) &
            (u2 >= 0) & (u2 < d2.shape[1]) & (v2 >= 0) & (v2 < d2.shape[0])
        )
        z1_mm = d1[v1[valid], u1[valid]]
        z2_mm = d2[v2[valid], u2[valid]]
        vd = (z1_mm > 100) & (z1_mm < 5000) & (z2_mm > 100) & (z2_mm < 5000)
        c_v = corrs[valid][vd]
        if len(c_v) < 8:
            continue
        z1 = z1_mm[vd].astype(np.float64) / 1000.0
        z2 = z2_mm[vd].astype(np.float64) / 1000.0
        X1 = (K_d_inv @ np.hstack([c_v[:, :2], np.ones((len(c_v), 1))]).T).T * z1[:, None]
        X2 = (K_d_inv @ np.hstack([c_v[:, 2:4], np.ones((len(c_v), 1))]).T).T * z2[:, None]
        pooled_c.append(c_v)
        pooled_X1.append(X1)
        pooled_X2.append(X2)

    if not pooled_X1:
        return None
    all_c = np.vstack(pooled_c)
    all_X1 = np.vstack(pooled_X1)
    all_X2 = np.vstack(pooled_X2)

    # Refine 3D-3D ground truth (R_3d, t_3d) on consistent depth correspondences
    R_init, t_init = p["R"].copy(), p["t"].flatten().copy()
    for th_m in (0.12, 0.06, 0.035):
        pred = (all_X1 @ R_init.T) + t_init.reshape(1, 3)
        mask = np.linalg.norm(pred - all_X2, axis=1) < th_m
        if np.sum(mask) < 12:
            break
        X_in, Y_in = all_X1[mask], all_X2[mask]
        cX, cY = X_in.mean(axis=0), Y_in.mean(axis=0)
        U, _, Vt = np.linalg.svd((X_in - cX).T @ (Y_in - cY))
        R_init = Vt.T @ U.T
        if np.linalg.det(R_init) < 0:
            Vt[-1, :] *= -1
            R_init = Vt.T @ U.T
        t_init = cY - R_init @ cX
    R_3d, t_3d = R_init, t_init

    # Refine 2D-3D ground truth (R_pnp, t_pnp) on consistent 2D-3D correspondences
    ok_pnp, rvec_p, tvec_p, inl_p = cv2.solvePnPRansac(
        all_X1.astype(np.float32),
        all_c[:, 2:4].astype(np.float32),
        K_rgb,
        None,
        iterationsCount=5000,
        reprojectionError=3.0,
        confidence=0.9999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if ok_pnp and inl_p is not None and len(inl_p) >= 12:
        idx_p = inl_p.ravel()
        rvec_p, tvec_p = cv2.solvePnPRefineLM(
            all_X1[idx_p].astype(np.float32),
            all_c[idx_p, 2:4].astype(np.float32),
            K_rgb,
            None,
            rvec_p,
            tvec_p,
        )
        R_pnp, _ = cv2.Rodrigues(rvec_p)
        t_pnp = tvec_p.flatten()
    else:
        R_pnp, t_pnp = R_3d, t_3d

    return d1, d2, K_d_inv, K_rgb, R_pnp, t_pnp, R_3d, t_3d


def load_7scenes_pnp_problems(data_dir=DEFAULT_DATA_DIR, detectors=("LoMa", "RoMa", "DISK+LG", "RootSIFT")):
    """Build real-world 2D-3D PnP problems from 7Scenes RGB-D image pairs (including high-outlier clutter regimes)."""
    pairs = load_7scenes_pairs(data_dir, include_regimes=False)
    if not pairs:
        return []
    cache = ensure_cached_matches(pairs, detectors, data_dir=data_dir)
    problems = []
    rng = np.random.default_rng(7070)
    for p in pairs:
        calib = _calibrate_7scenes_rgbd_pair(p, cache, detectors)
        if calib is None:
            continue
        d1, _, K_d_inv, K2, R_gt, t_gt, _, _ = calib
        h2, w2 = float(p["h2"]), float(p["w2"])
        for det in detectors:
            det_key = canonical_detector_name(det)
            corrs, scores = cache[p["cache_key"]][det_key]
            if len(corrs) < 20:
                continue
            u1 = np.round(corrs[:, 0]).astype(int)
            v1 = np.round(corrs[:, 1]).astype(int)
            valid = (u1 >= 0) & (u1 < d1.shape[1]) & (v1 >= 0) & (v1 < d1.shape[0])
            z1_mm = d1[v1[valid], u1[valid]]
            valid_d = (z1_mm > 100) & (z1_mm < 5000)
            c_v = corrs[valid][valid_d]
            s_v = scores[valid][valid_d]
            z1 = z1_mm[valid_d].astype(np.float64) / 1000.0
            if len(c_v) < 20:
                continue
            uv1_h = np.hstack([c_v[:, :2], np.ones((len(c_v), 1))])
            X1 = (K_d_inv @ uv1_h.T).T * z1[:, None]
            pts2d = np.ascontiguousarray(c_v[:, 2:4], dtype=np.float64)
            pts3d = np.ascontiguousarray(X1, dtype=np.float64)

            # Filter consistent vs background points to create realistic high-outlier RGB-D PnP problem
            X_cam = (pts3d @ R_gt.T) + t_gt.reshape(1, 3)
            uv_proj = (X_cam @ K2.T)
            uv_proj = uv_proj[:, :2] / (uv_proj[:, 2:3] + 1e-12)
            reproj = np.linalg.norm(uv_proj - pts2d, axis=1)
            inl_mask = (reproj < 3.5) & (X_cam[:, 2] > 0.1)
            if np.sum(inl_mask) < 12:
                continue
            in_2d, in_3d, in_s = pts2d[inl_mask], pts3d[inl_mask], s_v[inl_mask]
            target_in = min(len(in_2d), 65)
            in_2d, in_3d, in_s = in_2d[:target_in], in_3d[:target_in], in_s[:target_in]

            # Add cluttered 2D-3D outliers (4x to 7x inliers)
            n_out = int(rng.integers(3 * target_in, 7 * target_in + 1))
            mn3, mx3 = pts3d.min(axis=0), pts3d.max(axis=0)
            out_2d = np.column_stack([rng.uniform(0, w2, n_out), rng.uniform(0, h2, n_out)])
            out_3d = rng.uniform(mn3, mx3, size=(n_out, 3))
            all_2d = np.vstack([in_2d, out_2d])
            all_3d = np.vstack([in_3d, out_3d])
            pnp_corrs = np.ascontiguousarray(np.hstack([all_2d, all_3d]), dtype=np.float64)
            perm = rng.permutation(len(pnp_corrs))
            pnp_corrs = np.ascontiguousarray(pnp_corrs[perm], dtype=np.float64)

            problems.append({
                "dataset": "7Scenes RGB-D",
                "detector": det_key,
                "scene": p["scene"],
                "corrs": pnp_corrs,
                "probs": np.zeros(0, dtype=np.float64),
                "K": K2,
                "R_gt": R_gt,
                "t_gt": t_gt,
                "width": w2,
                "height": h2,
                "threshold_px": 3.5,
                "cm_scale": 100.0,
                "comb_trans_weight": 10.0 / 50.0,
            })
    return problems


def _random_rotation_matrix(rng, max_angle_deg=90.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.deg2rad(rng.uniform(5.0, max_angle_deg))
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def load_pose6dscene_pnp_problems(data_dir=DEFAULT_DATA_DIR, n_trials_per_regime=20):
    """Load calibrated 2D-3D PnP benchmark problems from pose6dscene across 7 outlier regimes (33% to 91%)."""
    ex_data_dir = os.path.join(SUPERANSAC_DIR, "examples", "data")
    pts_path = os.path.join(ex_data_dir, "pose6dscene_points.txt")
    gt_path = os.path.join(ex_data_dir, "pose6dscene_gt.txt")
    k_path = os.path.join(ex_data_dir, "pose6dscene.K")
    if not (os.path.exists(pts_path) and os.path.exists(gt_path) and os.path.exists(k_path)):
        return []

    corrs_gt = np.loadtxt(pts_path, dtype=np.float64).copy()
    gt_pose = np.loadtxt(gt_path, dtype=np.float64)
    K = np.loadtxt(k_path, dtype=np.float64)
    R_gt = gt_pose[:, :3].copy()
    t_gt = gt_pose[:, 3].copy()

    min_gt = corrs_gt.min(axis=0)
    max_gt = corrs_gt.max(axis=0)
    w = float(max(K[0, 2] * 2.0, max_gt[0] + 20.0))
    h = float(max(K[1, 2] * 2.0, max_gt[1] + 20.0))

    outlier_mults = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    outlier_labels = ["33%", "50%", "67%", "75%", "83%", "88%", "91%"]
    problems = []

    for m_idx, (mult, lbl) in enumerate(zip(outlier_mults, outlier_labels)):
        rng = np.random.default_rng(2026 + m_idx * 37)
        n_in = len(corrs_gt)
        n_out = int(round(mult * n_in))
        for trial in range(n_trials_per_regime):
            outliers = rng.uniform(min_gt, max_gt, size=(n_out, 5))
            pts = np.vstack([corrs_gt, outliers])
            pts[:, :2] += rng.normal(0.0, 1.5, size=(len(pts), 2))
            corrs = np.ascontiguousarray(pts, dtype=np.float64)
            problems.append({
                "dataset": "pose6dscene",
                "detector": f"outlier_{lbl}",
                "scene": f"pose6d_{lbl}_{trial}",
                "corrs": corrs,
                "probs": np.zeros(0, dtype=np.float64),
                "K": K,
                "R_gt": R_gt,
                "t_gt": t_gt,
                "width": w,
                "height": h,
                "threshold_px": 3.5,
                "cm_scale": 1.0,  # scene units are already cm
                "comb_trans_weight": 10.0 / 50.0,
            })
    return problems


def load_7scenes_rigid_problems(data_dir=DEFAULT_DATA_DIR, detectors=("LoMa", "RoMa", "DISK+LG", "RootSIFT")):
    """Build real-world 3D-3D rigid registration problems from 7Scenes RGB-D image pairs."""
    pairs = load_7scenes_pairs(data_dir, include_regimes=False)
    if not pairs:
        return []
    cache = ensure_cached_matches(pairs, detectors, data_dir=data_dir)
    problems = []
    rng = np.random.default_rng(8080)
    for p in pairs:
        calib = _calibrate_7scenes_rgbd_pair(p, cache, detectors)
        if calib is None:
            continue
        d1, d2, K_d_inv, _, _, _, R_gt, t_gt = calib
        for det in detectors:
            det_key = canonical_detector_name(det)
            corrs, scores = cache[p["cache_key"]][det_key]
            if len(corrs) < 20:
                continue
            u1 = np.round(corrs[:, 0]).astype(int)
            v1 = np.round(corrs[:, 1]).astype(int)
            u2 = np.round(corrs[:, 2]).astype(int)
            v2 = np.round(corrs[:, 3]).astype(int)
            valid = (
                (u1 >= 0) & (u1 < d1.shape[1]) & (v1 >= 0) & (v1 < d1.shape[0]) &
                (u2 >= 0) & (u2 < d2.shape[1]) & (v2 >= 0) & (v2 < d2.shape[0])
            )
            z1_mm = d1[v1[valid], u1[valid]]
            z2_mm = d2[v2[valid], u2[valid]]
            valid_d = (z1_mm > 100) & (z1_mm < 5000) & (z2_mm > 100) & (z2_mm < 5000)
            c_v = corrs[valid][valid_d]
            s_v = scores[valid][valid_d]
            z1 = z1_mm[valid_d].astype(np.float64) / 1000.0
            z2 = z2_mm[valid_d].astype(np.float64) / 1000.0
            if len(c_v) < 20:
                continue
            uv1_h = np.hstack([c_v[:, :2], np.ones((len(c_v), 1))])
            uv2_h = np.hstack([c_v[:, 2:4], np.ones((len(c_v), 1))])
            X1 = (K_d_inv @ uv1_h.T).T * z1[:, None]
            X2 = (K_d_inv @ uv2_h.T).T * z2[:, None]
            pred2 = (X1 @ R_gt.T) + t_gt.reshape(1, 3)
            res3d = np.linalg.norm(pred2 - X2, axis=1)
            inl_mask = res3d < 0.045
            if np.sum(inl_mask) < 12:
                continue
            in_X1, in_X2 = X1[inl_mask], X2[inl_mask]
            target_in = min(len(in_X1), 65)
            in_X1, in_X2 = in_X1[:target_in], in_X2[:target_in]

            # Add 3D clutter outliers (4x to 8x inliers)
            n_out = int(rng.integers(4 * target_in, 8 * target_in + 1))
            mn1, mx1 = X1.min(axis=0) - 0.3, X1.max(axis=0) + 0.3
            mn2, mx2 = X2.min(axis=0) - 0.3, X2.max(axis=0) + 0.3
            out_X1 = rng.uniform(mn1, mx1, size=(n_out, 3))
            out_X2 = rng.uniform(mn2, mx2, size=(n_out, 3))
            rigid_corrs = np.vstack([np.hstack([in_X1, in_X2]), np.hstack([out_X1, out_X2])])
            perm = rng.permutation(len(rigid_corrs))
            rigid_corrs = np.ascontiguousarray(rigid_corrs[perm], dtype=np.float64)
            probs = np.ascontiguousarray(np.linspace(1.0, 0.01, len(rigid_corrs)), dtype=np.float64)
            problems.append({
                "dataset": "7Scenes RGB-D",
                "detector": det_key,
                "scene": p["scene"],
                "corrs": rigid_corrs,
                "probs": probs,
                "R_gt": R_gt,
                "t_gt": t_gt,
                "threshold_3d": 0.20,
                "cm_scale": 100.0,
                "comb_trans_weight": 0.5,
            })
    return problems


def load_rigid_pose_example_problems(data_dir=DEFAULT_DATA_DIR, n_trials_per_tier=25):
    """Load 3D-3D point cloud registration benchmark problems across 4 difficulty tiers."""
    ex_data_dir = os.path.join(SUPERANSAC_DIR, "examples", "data")
    pts_path = os.path.join(ex_data_dir, "rigid_pose_example_points.txt")
    gt_path = os.path.join(ex_data_dir, "rigid_pose_example_gt.txt")
    p6d_path = os.path.join(ex_data_dir, "pose6dscene_points.txt")
    if not (os.path.exists(pts_path) and os.path.exists(gt_path)):
        return []

    rng = np.random.default_rng(4004)
    real_corrs = np.loadtxt(pts_path, dtype=np.float64)
    real_gt_raw = np.loadtxt(gt_path, dtype=np.float64)[:4, :]

    p1_h = np.hstack([real_corrs[:, :3], np.ones((len(real_corrs), 1))])
    p2_gt = (p1_h @ real_gt_raw.T)[:, :3]
    res_gt = np.linalg.norm(p2_gt - real_corrs[:, 3:6], axis=1)
    tight_inliers = real_corrs[res_gt < 0.045]
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

    pose6d_pts = np.loadtxt(p6d_path, dtype=np.float64)[:, 2:5] if os.path.exists(p6d_path) else real_corrs[:, :3]
    pose6d_pts = (pose6d_pts - np.mean(pose6d_pts, axis=0)) * 0.02
    base_clouds = [real_corrs[:, :3], pose6d_pts]

    problems = []
    # Tier 1: Real scan subsampled pairs (8% - 22% inliers)
    for i in range(n_trials_per_tier):
        n_in = int(rng.integers(35, 90))
        n_out = int(rng.integers(250, 700))
        inl_pool = tight_inliers if (i % 2 == 0) else med_inliers
        idx_in = rng.choice(len(inl_pool), size=n_in, replace=False)
        in_pts = inl_pool[idx_in].copy()
        mn = np.min(real_corrs, axis=0) - 0.5
        mx = np.max(real_corrs, axis=0) + 0.5
        n_nat = min(len(natural_outliers), n_out // 2)
        idx_nat = rng.choice(len(natural_outliers), size=n_nat, replace=False)
        uni_out = rng.uniform(mn, mx, size=(n_out - n_nat, 6))
        corrs = np.vstack([in_pts, natural_outliers[idx_nat], uni_out])
        R_delta = _random_rotation_matrix(rng, max_angle_deg=45.0)
        t_delta = rng.uniform(-0.8, 0.8, size=3)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta
        corrs[:, 3:6] = (corrs[:, 3:6] @ R_delta.T) + t_delta.reshape(1, 3)
        T_gt = T_delta @ real_gt
        perm = rng.permutation(len(corrs))
        corrs = np.ascontiguousarray(corrs[perm], dtype=np.float64)
        probs = np.ascontiguousarray(np.linspace(1.0, 0.01, len(corrs)), dtype=np.float64)
        problems.append({
            "dataset": "rigid_pose_example",
            "detector": "tier1_real_scan",
            "scene": f"real_scan_t1_{i:02d}",
            "corrs": corrs,
            "probs": probs,
            "R_gt": T_gt[:3, :3],
            "t_gt": T_gt[:3, 3],
            "threshold_3d": 0.20,
            "cm_scale": 100.0,
            "comb_trans_weight": 0.5,
        })

    # Tier 2: High-outlier (75% - 92% outliers) real scan registration
    for i in range(n_trials_per_tier):
        n_in = int(rng.integers(30, 80))
        n_out = int(rng.integers(250, 750))
        idx_in = rng.choice(len(tight_inliers), size=n_in, replace=False)
        in_pts = tight_inliers[idx_in].copy()
        mn = np.min(real_corrs, axis=0) - 0.5
        mx = np.max(real_corrs, axis=0) + 0.5
        uni_out = rng.uniform(mn, mx, size=(n_out, 6))
        corrs = np.vstack([in_pts, uni_out])
        R_delta = _random_rotation_matrix(rng, max_angle_deg=90.0)
        t_delta = rng.uniform(-1.5, 1.5, size=3)
        T_delta = np.eye(4)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta
        corrs[:, 3:6] = (corrs[:, 3:6] @ R_delta.T) + t_delta.reshape(1, 3)
        T_gt = T_delta @ real_gt
        perm = rng.permutation(len(corrs))
        corrs = np.ascontiguousarray(corrs[perm], dtype=np.float64)
        probs = np.ascontiguousarray(np.linspace(1.0, 0.01, len(corrs)), dtype=np.float64)
        problems.append({
            "dataset": "rigid_pose_example",
            "detector": "tier2_high_outlier",
            "scene": f"real_scan_t2_{i:02d}",
            "corrs": corrs,
            "probs": probs,
            "R_gt": T_gt[:3, :3],
            "t_gt": T_gt[:3, 3],
            "threshold_3d": 0.20,
            "cm_scale": 100.0,
            "comb_trans_weight": 0.5,
        })

    # Tier 3: Real 3D point cloud registration with sensor noise + outliers
    for i in range(n_trials_per_tier):
        cloud = base_clouds[i % 2]
        n_in = int(rng.integers(45, 120))
        n_out = int(rng.integers(200, 600))
        idx = rng.choice(len(cloud), size=n_in, replace=(n_in > len(cloud)))
        X = cloud[idx].copy()
        R_gt = _random_rotation_matrix(rng, max_angle_deg=120.0)
        t_gt = rng.uniform(-2.0, 2.0, size=3)
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
        corrs = np.ascontiguousarray(corrs[ord_idx], dtype=np.float64)
        probs = np.ascontiguousarray(sc[ord_idx], dtype=np.float64)
        problems.append({
            "dataset": "rigid_pose_example",
            "detector": "tier3_sensor_noise",
            "scene": f"real_cloud_t3_{i:02d}",
            "corrs": corrs,
            "probs": probs,
            "R_gt": R_gt,
            "t_gt": t_gt,
            "threshold_3d": 0.20,
            "cm_scale": 100.0,
            "comb_trans_weight": 0.5,
        })

    # Tier 4: Extreme outlier ratio (85% - 94% outliers) & surface mismatch clutter
    for i in range(n_trials_per_tier):
        cloud = base_clouds[i % 2]
        n_in = int(rng.integers(25, 60))
        n_out = int(rng.integers(300, 700))
        idx = rng.choice(len(cloud), size=n_in, replace=False)
        X = cloud[idx].copy()
        R_gt = _random_rotation_matrix(rng, max_angle_deg=150.0)
        t_gt = rng.uniform(-2.5, 2.5, size=3)
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
        corrs = np.ascontiguousarray(corrs[perm], dtype=np.float64)
        probs = np.ascontiguousarray(np.linspace(1.0, 0.01, len(corrs)), dtype=np.float64)
        problems.append({
            "dataset": "rigid_pose_example",
            "detector": "tier4_extreme",
            "scene": f"real_surface_mismatch_t4_{i:02d}",
            "corrs": corrs,
            "probs": probs,
            "R_gt": R_gt,
            "t_gt": t_gt,
            "threshold_3d": float(rng.uniform(0.18, 0.22)),
            "cm_scale": 100.0,
            "comb_trans_weight": 0.5,
        })
    return problems
