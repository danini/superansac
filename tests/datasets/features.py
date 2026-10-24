"""Feature extraction and match cache shared by all SupeRANSAC benchmark scripts.

Supported matchers: LoMa, RoMa, LoFTR, DISK+LightGlue, RootSIFT, SIFT, ORB.
Each cache entry is (corrs[N,4] float64 sorted by descending score, scores[N]).
Matches are the raw detector output: nothing is filtered, re-ordered, or injected.
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


def ensure_cached_matches(pairs, detectors, data_dir=DEFAULT_DATA_DIR, max_workers=12):
    """Ensure (pair['cache_key'], detector) entries exist for every pair and return the cache."""
    detectors = [canonical_detector_name(d) for d in detectors]
    cache = load_cache(data_dir)
    tasks = {}
    for p in pairs:
        k = p["cache_key"]
        existing = cache.get(k, {})
        needed = [d for d in detectors if d not in existing]
        if needed and k not in tasks:
            tasks[k] = (k, p["im1"], p["im2"], needed)

    if tasks:
        task_list = list(tasks.values())
        n_workers = min(max_workers, len(task_list))
        print(f"Extracting {detectors} features for {len(task_list)} image pairs using {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, (k, out) in enumerate(ex.map(_extract_worker, task_list), 1):
                cache.setdefault(k, {}).update(out)
                if idx % 10 == 0 or idx == len(task_list):
                    print(f"  Extracted {idx}/{len(task_list)} pairs...")
                    save_cache(cache, data_dir)
    return cache
