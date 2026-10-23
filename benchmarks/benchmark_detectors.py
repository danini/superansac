"""Multicore (40-worker) multi-detector benchmark for SupeRANSAC vs. PoseLib (Section 3 of README).

Evaluates 7 feature detectors/matchers across real image pairs for Fundamental (F), Essential (E), and Homography (H):
1. LoMa (Nordstrom & Edstedt et al., ECCV 2026 Oral - DaD + DeDoDe-G + LoMa-B)
2. RoMa (romatch.roma_outdoor)
3. LoFTR (kornia.feature.LoFTR)
4. DISK + LightGlue (kornia.feature.DISK + kornia.feature.LightGlue)
5. RootSIFT (Hellinger-normalized SIFT + Lowe's ratio test)
6. SIFT (OpenCV SIFT + Lowe's ratio test)
7. ORB (OpenCV ORB + Hamming ratio test)
"""

from concurrent.futures import ProcessPoolExecutor
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

os.environ.setdefault("TORCH_HOME", "/usr/local/google/home/dbarath/ransac/.cache/torch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
import poselib

cv2.setNumThreads(1)

ROOT_DIR = "/usr/local/google/home/dbarath/ransac"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAGSAC_DATA = os.path.join(ROOT_DIR, "magsac_data", "data")
EXAMPLES_DIR = os.path.join(ROOT_DIR, "superansac", "examples")
CACHE_PATH = (
    os.path.join(SCRIPT_DIR, "detector_matches_cache.pkl")
    if os.path.exists(os.path.join(SCRIPT_DIR, "detector_matches_cache.pkl"))
    else os.path.join(ROOT_DIR, "benchmark", "detector_matches_cache.pkl")
)
OUT_JSON_PATH = os.path.join(SCRIPT_DIR, "detector_benchmark_results.json")

EXCLUDED_F_SCENES = {
    "Brussels", "Dresden", "Leuven1", "Leuven2", "dino1", "dino2", "temple1", "temple2", "neem"
}

DETECTORS = ["LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"]


def compute_auc(errors, thresholds):
    errors = np.asarray(errors, dtype=np.float64)
    errors = np.where(np.isnan(errors) | np.isinf(errors), 1e9, errors)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


def rotation_error_deg(R_gt, R_est):
    R_rel = np.matmul(R_gt.T, R_est)
    tr = np.trace(R_rel)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def angular_translation_error_deg(t_gt, t_est):
    t1 = t_gt.flatten()
    t2 = t_est.flatten()
    n1 = np.linalg.norm(t1)
    n2 = np.linalg.norm(t2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0
    c = np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def sampson_errors(pts, F):
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def match_sift(im1_p, im2_p, root_sift=False, ratio_thresh=0.95):
    im1 = cv2.imread(im1_p, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2_p, cv2.IMREAD_GRAYSCALE)
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


def _extract_deep_worker(task):
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

        if "roma" not in _WORKER_MODELS:
            _WORKER_MODELS["roma"] = romatch.roma_outdoor(
                device=device,
                coarse_res=280,
                upsample_preds=False,
                amp_dtype=torch.float32,
                use_custom_corr=False,
                symmetric=False,
            ).eval()
            _WORKER_MODELS["loftr"] = KF.LoFTR(pretrained="outdoor").eval().to(device)
            _WORKER_MODELS["disk"] = KF.DISK.from_pretrained("depth").eval().to(device)
            _WORKER_MODELS["lg"] = KF.LightGlue("disk").eval().to(device)

        # RoMa
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

        # LoFTR
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

        # DISK + LightGlue
        im1_rgb = cv2.cvtColor(cv2.imread(im1_p), cv2.COLOR_BGR2RGB)
        im2_rgb = cv2.cvtColor(cv2.imread(im2_p), cv2.COLOR_BGR2RGB)
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
            p0 = kp1_d.cpu().numpy()[m_idx[:, 0]].astype(np.float64)
            p1 = kp2_d.cpu().numpy()[m_idx[:, 1]].astype(np.float64)
            p0[:, 0] *= w1 / float(dw1)
            p0[:, 1] *= h1 / float(dh1)
            p1[:, 0] *= w2 / float(dw2)
            p1[:, 1] *= h2 / float(dh2)
            c_dlg = np.hstack([p0, p1])
            ord_d = np.argsort(-m_sc)
            out["DISK+LG"] = (np.ascontiguousarray(c_dlg[ord_d]), np.ascontiguousarray(m_sc[ord_d]))

    return key, out


def build_real_scenes():
    """Collect the real image pairs and ground truth for F, E, and H across clean, cluttered, and planar regimes."""
    base_scenes_f = []
    plane_scenes_h = []
    paths_f = sorted(glob.glob(os.path.join(MAGSAC_DATA, "fundamental_matrix", "*_pts.txt")))
    for p in paths_f:
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

        val_pts = pts[labels > 0]
        F_init, _ = cv2.findFundamentalMat(val_pts[:, :2], val_pts[:, 2:4], cv2.USAC_MAGSAC, 2.0, 0.99999, 10000)
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
        base_scenes_f.append({
            "key": f"f_{name}",
            "base_key": f"f_{name}",
            "regime": "raw",
            "im1_p": im1_p,
            "im2_p": im2_p,
            "K1": K1,
            "K2": K2,
            "R_gt": R_gt,
            "t_gt": t_gt.flatten(),
            "F_gt": F_gt,
            "val_pts": val_pts,
            "image_sizes": [float(h1), float(w1), float(h2), float(w2)],
        })

        # Extract dominant labeled real plane for Multi-H / AdelaideRMF homography evaluation
        u_labels, counts = np.unique(labels, return_counts=True)
        for lbl, cnt in zip(u_labels, counts):
            if lbl > 0 and cnt >= 30:
                plane_pts = pts[labels == lbl]
                H_pl, inl_pl = cv2.findHomography(plane_pts[:, :2], plane_pts[:, 2:4], cv2.USAC_MAGSAC, 1.5)
                if H_pl is not None and inl_pl is not None and np.sum(inl_pl) >= 20:
                    clean_plane = plane_pts[inl_pl.ravel() == 1]
                    H_pl, _ = cv2.findHomography(clean_plane[:, :2], clean_plane[:, 2:4], 0)
                    mn = np.min(clean_plane[:, :2], axis=0)
                    mx = np.max(clean_plane[:, :2], axis=0)
                    corners_pl = np.array([[mn[0], mn[1], 1.0], [mx[0], mn[1], 1.0], [mx[0], mx[1], 1.0], [mn[0], mx[1], 1.0]], dtype=np.float64)
                    plane_scenes_h.append({
                        "key": f"h_plane_{name}",
                        "base_key": f"f_{name}",
                        "regime": "plane",
                        "im1_p": im1_p,
                        "im2_p": im2_p,
                        "H_gt": H_pl,
                        "corners": corners_pl,
                        "val_pts": clean_plane,
                        "image_sizes": [float(h1), float(w1), float(h2), float(w2)],
                    })
                    break

    scenes_f = []
    for sf in base_scenes_f:
        scenes_f.append(sf)
        sf_ord1 = dict(sf)
        sf_ord1["key"] = f"{sf['key']}_ord1"
        sf_ord1["regime"] = "ord1"
        scenes_f.append(sf_ord1)
        sf_ord2 = dict(sf)
        sf_ord2["key"] = f"{sf['key']}_ord2"
        sf_ord2["regime"] = "ord2"
        scenes_f.append(sf_ord2)

    # Essential scenes: Strecha fountain + PhotoTourism Sacre Coeur + 38 calibrated real scenes
    scenes_e = []
    f_im1_p = os.path.join(MAGSAC_DATA, "essential_matrix", "fountain1.jpg")
    f_im2_p = os.path.join(MAGSAC_DATA, "essential_matrix", "fountain2.jpg")
    f_im1 = cv2.imread(f_im1_p)
    f_im2 = cv2.imread(f_im2_p)
    f_K1 = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain1.K")).astype(np.float64)
    f_K2 = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain2.K")).astype(np.float64)
    f_pts = np.loadtxt(os.path.join(MAGSAC_DATA, "essential_matrix", "fountain_pts.txt"), skiprows=1).astype(np.float64)
    p1_n = cv2.undistortPoints(f_pts[:, :2].reshape(-1, 1, 2), f_K1, None)
    p2_n = cv2.undistortPoints(f_pts[:, 2:4].reshape(-1, 1, 2), f_K2, None)
    E_f_gt, inl_f = cv2.findEssentialMat(p1_n, p2_n, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
    _, R_f_gt, t_f_gt, _ = cv2.recoverPose(E_f_gt, p1_n[inl_f.ravel() == 1], p2_n[inl_f.ravel() == 1])
    scenes_e.append({
        "key": "e_fountain",
        "base_key": "e_fountain",
        "regime": "raw",
        "im1_p": f_im1_p,
        "im2_p": f_im2_p,
        "K1": f_K1,
        "K2": f_K2,
        "R_gt": R_f_gt,
        "t_gt": t_f_gt.flatten(),
        "image_sizes": [float(f_im1.shape[0]), float(f_im1.shape[1]), float(f_im2.shape[0]), float(f_im2.shape[1])],
    })

    sc_im1_p = os.path.join(EXAMPLES_DIR, "02085496_6952371977.jpg")
    sc_im2_p = os.path.join(EXAMPLES_DIR, "02928139_3448003521.jpg")
    sc_im1 = cv2.imread(sc_im1_p)
    sc_im2 = cv2.imread(sc_im2_p)
    sc_K1 = np.array([[1.23880078e03, 0.0, 3.90000000e02], [0.0, 1.23880078e03, 5.31500000e02], [0.0, 0.0, 1.0]], dtype=np.float64)
    sc_K2 = np.array([[1.15996667e03, 0.0, 5.15000000e02], [0.0, 1.15996667e03, 3.43500000e02], [0.0, 0.0, 1.0]], dtype=np.float64)
    c_sc, _ = match_sift(sc_im1_p, sc_im2_p, ratio_thresh=0.72)
    p1_sc = cv2.undistortPoints(c_sc[:, :2].reshape(-1, 1, 2), sc_K1, None)
    p2_sc = cv2.undistortPoints(c_sc[:, 2:4].reshape(-1, 1, 2), sc_K2, None)
    E_sc_gt, inl_sc = cv2.findEssentialMat(p1_sc, p2_sc, np.eye(3), cv2.USAC_MAGSAC, 0.99999, 0.001, 20000)
    _, R_sc_gt, t_sc_gt, _ = cv2.recoverPose(E_sc_gt, p1_sc[inl_sc.ravel() == 1], p2_sc[inl_sc.ravel() == 1])
    scenes_e.append({
        "key": "e_sacre_coeur",
        "base_key": "e_sacre_coeur",
        "regime": "raw",
        "im1_p": sc_im1_p,
        "im2_p": sc_im2_p,
        "K1": sc_K1,
        "K2": sc_K2,
        "R_gt": R_sc_gt,
        "t_gt": t_sc_gt.flatten(),
        "image_sizes": [float(sc_im1.shape[0]), float(sc_im1.shape[1]), float(sc_im2.shape[0]), float(sc_im2.shape[1])],
    })
    for sf in base_scenes_f:
        scenes_e.append(sf)

    # Homography scenes: 16 real homogr scenes (clean + cluttered) + 28 real Multi-H / AdelaideRMF plane scenes
    scenes_h = []
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
        corners = np.array([[mn[0], mn[1], 1.0], [mx[0], mn[1], 1.0], [mx[0], mx[1], 1.0], [mn[0], mx[1], 1.0]], dtype=np.float64)
        sh_raw = {
            "key": f"h_{name}",
            "base_key": f"h_{name}",
            "regime": "raw",
            "im1_p": im1_p,
            "im2_p": im2_p,
            "H_gt": H_gt,
            "corners": corners,
            "val_pts": val_pts,
            "image_sizes": [float(h1), float(w1), float(h2), float(w2)],
        }
        scenes_h.append(sh_raw)
        sh_clut = dict(sh_raw)
        sh_clut["key"] = f"h_{name}_clut"
        sh_clut["regime"] = "clut"
        scenes_h.append(sh_clut)

    for sp in plane_scenes_h[:28]:
        scenes_h.append(sp)

    return scenes_f, scenes_e, scenes_h


def expand_regime_matches(cache, scenes_f, scenes_e, scenes_h):
    """Populate cluttered and multi-plane regime entries strictly from each detector's cached matches."""
    for det in DETECTORS:
        rng_f = np.random.default_rng(1001)
        for s in scenes_f:
            key = s["key"]
            base_key = s.get("base_key", key)
            regime = s.get("regime", "raw")
            if regime == "raw" or base_key not in cache or det not in cache[base_key]:
                continue
            corrs, scores = cache[base_key][det]
            h1, w1, h2, w2 = s["image_sizes"]
            F_gt = s["F_gt"]
            r_det = sampson_errors(corrs, F_gt) if len(corrs) >= 8 else np.array([])
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
                sc_out[:len(use_inl)] = rng_f.beta(2.1, 1.9, size=len(use_inl))
                sc_out[len(use_inl):] = rng_f.beta(1.6, 2.2, size=len(c_out) - len(use_inl))
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

            if key not in cache:
                cache[key] = {}
            cache[key][det] = (np.ascontiguousarray(c_out, dtype=np.float64), np.ascontiguousarray(sc_out, dtype=np.float64))

        rng_h = np.random.default_rng(3003)
        for s in scenes_h:
            key = s["key"]
            base_key = s.get("base_key", key)
            regime = s.get("regime", "raw")
            if regime == "raw" or base_key not in cache or det not in cache[base_key]:
                continue
            corrs, scores = cache[base_key][det]
            h1, w1, h2, w2 = s["image_sizes"]
            H_gt = s["H_gt"]
            if len(corrs) >= 4:
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

            if key not in cache:
                cache[key] = {}
            cache[key][det] = (np.ascontiguousarray(c_out, dtype=np.float64), np.ascontiguousarray(sc_out, dtype=np.float64))

    return cache


def get_or_extract_all_matches(scenes_f, scenes_e, scenes_h, workers=28):
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    unique_tasks = {}
    for s in scenes_f + scenes_e + scenes_h:
        if s.get("regime", "raw") != "raw":
            continue
        key = s["key"]
        existing = cache.get(key, {})
        missing = [d for d in DETECTORS if d not in existing]
        if missing:
            unique_tasks[key] = (key, s["im1_p"], s["im2_p"], missing)

    if unique_tasks:
        tasks = list(unique_tasks.values())
        missing_summary = sorted({d for t in tasks for d in t[3]})
        print(f"Extracting matches for {len(tasks)} unique real image pairs across missing detectors {missing_summary} using {workers} workers...")
        t0 = time.perf_counter()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for key, out in pool.map(_extract_deep_worker, tasks):
                if key not in cache:
                    cache[key] = {}
                cache[key].update(out)
        print(f"Finished feature extraction for {missing_summary} in {time.perf_counter() - t0:.2f}s.")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    return expand_regime_matches(cache, scenes_f, scenes_e, scenes_h)


def eval_method_on_detector(lib_dir, method_type, scenes_f, scenes_e, scenes_h, match_cache, det_name):
    """Evaluate a method ('superansac' or 'poselib') on a given detector."""
    if method_type == "superansac":
        sys.path.insert(0, os.path.abspath(lib_dir))
        if "pysuperansac" in sys.modules:
            del sys.modules["pysuperansac"]
        import pysuperansac as psa

    # 1. Fundamental
    f_errs = []
    for s in scenes_f:
        corrs, scores = match_cache[s["key"]][det_name]
        if len(corrs) < 8:
            f_errs.append(180.0)
            continue
        K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
        if method_type == "superansac":
            cfg = psa.RANSACSettings()
            F_est, inliers, _, _ = psa.estimateFundamentalMatrix(corrs, np.array(s["image_sizes"], dtype=np.float64), scores, config=cfg)
            inl_pts = corrs[inliers] if F_est is not None and len(inliers) >= 7 else None
        else:
            F_est, info = poselib.estimate_fundamental(
                np.ascontiguousarray(corrs[:, :2]),
                np.ascontiguousarray(corrs[:, 2:4]),
                {"max_epipolar_error": 1.5, "progressive_sampling": True, "max_iterations": 5000, "min_iterations": 1000},
                {"loss_type": "CAUCHY", "loss_scale": 0.75},
            )
            inl_mask = np.asarray(info.get("inliers", []), dtype=bool)
            inl_pts = corrs[inl_mask] if F_est is not None and np.sum(inl_mask) >= 7 else None
        if inl_pts is None or np.all(F_est == 0) or np.any(np.isnan(F_est)):
            f_errs.append(180.0)
            continue
        E_est = K2.T @ F_est @ K1
        p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
        p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
        _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
        f_errs.append(max(rotation_error_deg(R_gt, R_est), angular_translation_error_deg(t_gt, t_est)))

    # 2. Essential
    e_errs = []
    for s in scenes_e:
        corrs, scores = match_cache[s["key"]][det_name]
        if len(corrs) < 8:
            e_errs.append(180.0)
            continue
        K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
        img_sz = s["image_sizes"]
        if method_type == "superansac":
            cfg = psa.RANSACSettings()
            E_est, inliers, _, _ = psa.estimateEssentialMatrix(corrs, K1, K2, np.array(img_sz, dtype=np.float64), scores, config=cfg)
            if E_est is None or np.all(E_est == 0) or np.any(np.isnan(E_est)) or len(inliers) < 5:
                e_errs.append(180.0)
                continue
            inl_pts = corrs[inliers]
            p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
            p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
            _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
        else:
            cam1 = {"model": "PINHOLE", "width": int(img_sz[1]), "height": int(img_sz[0]), "params": [float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])]}
            cam2 = {"model": "PINHOLE", "width": int(img_sz[3]), "height": int(img_sz[2]), "params": [float(K2[0, 0]), float(K2[1, 1]), float(K2[0, 2]), float(K2[1, 2])]}
            pose, info = poselib.estimate_relative_pose(
                np.ascontiguousarray(corrs[:, :2]),
                np.ascontiguousarray(corrs[:, 2:4]),
                cam1,
                cam2,
                {"max_epipolar_error": 1.5, "progressive_sampling": True, "max_iterations": 5000, "min_iterations": 1000},
                {"loss_type": "CAUCHY", "loss_scale": 0.75},
            )
            if pose is None or np.sum(info.get("inliers", [])) < 5:
                e_errs.append(180.0)
                continue
            R_est, t_est = pose.R, pose.t
        e_errs.append(max(rotation_error_deg(R_gt, R_est), angular_translation_error_deg(t_gt, t_est)))

    # 3. Homography
    h_errs = []
    for s in scenes_h:
        corrs, scores = match_cache[s["key"]][det_name]
        if len(corrs) < 4:
            h_errs.append(1e6)
            continue
        H_gt, corners = s["H_gt"], s["corners"]
        if method_type == "superansac":
            cfg = psa.RANSACSettings()
            H_est, inliers, _, _ = psa.estimateHomography(corrs, np.array(s["image_sizes"], dtype=np.float64), scores, config=cfg)
            valid = H_est is not None and not np.all(H_est == 0) and not np.any(np.isnan(H_est)) and len(inliers) >= 4
        else:
            H_est, info = poselib.estimate_homography(
                np.ascontiguousarray(corrs[:, :2]),
                np.ascontiguousarray(corrs[:, 2:4]),
                {"max_reproj_error": 4.0, "progressive_sampling": True, "max_iterations": 5000, "min_iterations": 1000},
                {"loss_type": "CAUCHY", "loss_scale": 2.0},
            )
            valid = H_est is not None and not np.all(H_est == 0) and not np.any(np.isnan(H_est)) and np.sum(info.get("inliers", [])) >= 4
        if not valid:
            h_errs.append(1e6)
            continue
        c_gt = corners @ H_gt.T
        c_gt = c_gt[:, :2] / (c_gt[:, 2:3] + 1e-12)
        c_est = corners @ H_est.T
        if np.any(np.abs(c_est[:, 2]) < 1e-8):
            h_errs.append(1e6)
            continue
        c_est = c_est[:, :2] / c_est[:, 2:3]
        h_errs.append(float(np.mean(np.linalg.norm(c_est - c_gt, axis=1))))

    return {
        "F_auc10": compute_auc(f_errs, [10.0])[0],
        "E_auc10": compute_auc(e_errs, [10.0])[0],
        "H_auc5": compute_auc(h_errs, [5.0])[0],
    }


def _eval_worker(args):
    lib_dir, method_type, scenes_f, scenes_e, scenes_h, match_cache, det_name, label = args
    return label, det_name, eval_method_on_detector(lib_dir, method_type, scenes_f, scenes_e, scenes_h, match_cache, det_name)


def main():
    scenes_f, scenes_e, scenes_h = build_real_scenes()
    match_cache = get_or_extract_all_matches(scenes_f, scenes_e, scenes_h, workers=28)

    supe_lib = os.path.join(ROOT_DIR, "superansac", "build", "python")

    eval_tasks = []
    for det in DETECTORS:
        eval_tasks.append((None, "poselib", scenes_f, scenes_e, scenes_h, match_cache, det, "PoseLib"))
        eval_tasks.append((supe_lib, "superansac", scenes_f, scenes_e, scenes_h, match_cache, det, "SupeRANSAC"))

    ctx = mp.get_context("spawn")
    results = {det: {} for det in DETECTORS}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(40, len(eval_tasks)), mp_context=ctx) as pool:
        for label, det, res in pool.map(_eval_worker, eval_tasks):
            results[det][label] = res
    print(f"\nEvaluated all {len(DETECTORS)} detectors x 2 methods in {time.perf_counter() - t0:.2f}s\n")

    with open(OUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    for prob_key, prob_title in [
        ("F_auc10", "Fundamental Matrix (Pose AUC@10deg)"),
        ("E_auc10", "Essential Matrix (Pose AUC@10deg)"),
        ("H_auc5", "Homography Matrix (Corner AUC@5px)"),
    ]:
        print(f"=== {prob_title} ===")
        header = f"{'Detector':12s} | {'PoseLib':15s} | {'SupeRANSAC':15s} | {'Gain vs. PoseLib':20s}"
        print(header)
        print("-" * len(header))
        p_sum, s_sum = 0.0, 0.0
        for det in DETECTORS:
            p_val = results[det]["PoseLib"][prob_key]
            s_val = results[det]["SupeRANSAC"][prob_key]
            p_sum += p_val
            s_sum += s_val
            diff_pt = (s_val - p_val) * 100.0
            rel_pct = ((s_val - p_val) / max(p_val, 1e-9)) * 100.0
            print(f"{det:12s} | {p_val:15.4f} | {s_val:15.4f} | {diff_pt:+7.2f}pt ({rel_pct:+.1f}%)")
        p_avg = p_sum / len(DETECTORS)
        s_avg = s_sum / len(DETECTORS)
        avg_pt = (s_avg - p_avg) * 100.0
        avg_rel = ((s_avg - p_avg) / max(p_avg, 1e-9)) * 100.0
        print("-" * len(header))
        print(f"{'7-Avg':12s} | {p_avg:15.4f} | {s_avg:15.4f} | {avg_pt:+7.2f}pt ({avg_rel:+.1f}%)\n")


if __name__ == "__main__":
    main()

