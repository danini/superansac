"""Generate benchmark plots and tables for Sections 1, 2, 3, and 4 of README.md."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import poselib

cv2.setNumThreads(1)

ROOT = "/usr/local/google/home/dbarath/ransac"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from benchmark_detectors import (
    build_real_scenes,
    compute_auc,
    get_or_extract_all_matches,
    rotation_error_deg,
    angular_translation_error_deg,
)

ASSETS_DIR = os.path.join(ROOT, "superansac", "assets")
CACHE_PATH = (
    os.path.join(SCRIPT_DIR, "detector_matches_cache.pkl")
    if os.path.exists(os.path.join(SCRIPT_DIR, "detector_matches_cache.pkl"))
    else os.path.join(ROOT, "benchmark", "detector_matches_cache.pkl")
)
BENCHMARK_DATA_PATH = (
    os.path.join(SCRIPT_DIR, "benchmark_data.pkl")
    if os.path.exists(os.path.join(SCRIPT_DIR, "benchmark_data.pkl"))
    else os.path.join(ROOT, "benchmark", "benchmark_data.pkl")
)
SUPERANSAC_SO = os.path.join(ROOT, "superansac", "build", "python", "pysuperansac.cpython-313-x86_64-linux-gnu.so")

_MOD_CACHE = {}


def _get_mod(so_path):
    if so_path not in _MOD_CACHE:
        spec = importlib.util.spec_from_file_location("pysuperansac", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD_CACHE[so_path] = mod
    return _MOD_CACHE[so_path]


def make_like_colab(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#E6E6E6")
    ax.grid(color="white", linestyle="solid", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)


def plot_section1_overview():
    with open(os.path.join(SCRIPT_DIR, "final_results.json"), "r") as f:
        main_res = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "poselib_results.json"), "r") as f:
        pl_res = json.load(f)["best"]
    abspose_path = os.path.join(SCRIPT_DIR, "abspose_rigid_results.json")
    if os.path.exists(abspose_path):
        with open(abspose_path, "r") as f:
            ar_res = json.load(f)
        pnp_cv = ar_res["absolute_pose"]["opencv"]["auc_10deg_50cm"]
        pnp_su = ar_res["absolute_pose"]["superansac"]["auc_10deg_50cm"]
        rig_cv = ar_res["rigid_pose"]["opencv"]["pose_auc_5deg_10cm"]
        rig_su = ar_res["rigid_pose"]["superansac"]["pose_auc_5deg_10cm"]
    else:
        pnp_cv, pnp_su = 0.2001, 0.7336
        rig_cv, rig_su = 0.1666, main_res["rigid_pose"]["primary_auc"]

    c_pose = mpl.colormaps["tab10"](2)
    c_cv = mpl.colormaps["tab10"](0)
    c_supe = mpl.colormaps["tab10"](3)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    make_like_colab(fig, ax)

    problems = [
        "Fundamental ($F$)\nPose AUC@10°",
        "Essential ($E$)\nPose AUC@10°",
        "Homography ($H$)\nCorner AUC@5px",
        "6D Abs. Pose ($PnP$)\nAUC@(10°,50cm)",
        "6D Rigid ($SE(3)$)\nAUC@(5°,10cm)",
    ]
    baseline_vals = [
        pl_res["fundamental"]["primary_auc"],
        pl_res["essential"]["primary_auc"],
        pl_res["homography"]["primary_auc"],
        pnp_cv,
        rig_cv,
    ]
    supe_vals = [
        main_res["fundamental"]["primary_auc"],
        main_res["essential"]["primary_auc"],
        main_res["homography"]["primary_auc"],
        pnp_su,
        rig_su,
    ]

    x = np.arange(len(problems))
    w = 0.34

    b_pl = ax.bar(x[:3] - w / 2, baseline_vals[:3], width=w, label="LO-RANSAC [PoseLib]", color=c_pose, zorder=3)
    b_cv = ax.bar(x[3:] - w / 2, baseline_vals[3:], width=w, label="OpenCV RANSAC", color=c_cv, zorder=3)
    b_su = ax.bar(x + w / 2, supe_vals, width=w, label="SupeRANSAC", color=c_supe, zorder=3)

    for rect in list(b_pl) + list(b_cv):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.4f}", ha="center", va="bottom", fontsize=9.8, fontweight="bold")
    for rect in b_su:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.4f}", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#990000")

    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=11.0, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("AUC Score", fontsize=13.5, fontweight="bold")
    ax.set_title("Benchmark Overview: SupeRANSAC vs. PoseLib / OpenCV RANSAC", fontsize=14.5, fontweight="bold", pad=12)
    ax.legend(loc="upper center", ncol=3, fontsize=11.0, frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    out_p = os.path.join(ASSETS_DIR, "benchmark_overview.png")
    plt.savefig(out_p, dpi=180)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_section3_detectors():
    with open(os.path.join(SCRIPT_DIR, "detector_benchmark_results.json"), "r") as f:
        det_res = json.load(f)

    c_pose = mpl.colormaps["tab10"](2)
    c_supe = mpl.colormaps["tab10"](3)

    detectors = ["LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"]
    configs = [
        ("F_auc10", "Fundamental Matrix (Pose AUC@10°)", "AUC @ 10°", "detector_fundamental.png", 0.72),
        ("E_auc10", "Essential Matrix (Pose AUC@10°)", "AUC @ 10°", "detector_essential.png", 0.65),
        ("H_auc5", "Homography Matrix (Corner AUC@5px)", "Corner AUC @ 5px", "detector_homography.png", 0.72),
    ]

    fig_all, axes_all = plt.subplots(1, 3, figsize=(18.0, 5.2), dpi=180)
    for idx, (metric_key, title, ylabel, fname, ymax) in enumerate(configs):
        ax_sub = axes_all[idx]
        make_like_colab(fig_all, ax_sub)
        x = np.arange(len(detectors))
        w = 0.36
        v_pose = [det_res[d]["PoseLib"][metric_key] for d in detectors]
        v_supe = [det_res[d].get("SupeRANSAC", det_res[d].get("Impr_SupeRANSAC"))[metric_key] for d in detectors]

        b1 = ax_sub.bar(x - w / 2, v_pose, width=w, label="LO-RANSAC [PoseLib]", color=c_pose, zorder=3)
        b2 = ax_sub.bar(x + w / 2, v_supe, width=w, label="SupeRANSAC", color=c_supe, zorder=3)
        for rect in b1:
            h = rect.get_height()
            ax_sub.text(rect.get_x() + rect.get_width() / 2.0, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for rect in b2:
            h = rect.get_height()
            ax_sub.text(rect.get_x() + rect.get_width() / 2.0, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#990000")
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels(detectors, fontsize=10.5, fontweight="bold", rotation=20)
        dyn_ymax = max(ymax, max(max(v_pose), max(v_supe)) * 1.18)
        ax_sub.set_ylim(0.0, dyn_ymax)
        ax_sub.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax_sub.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax_sub.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", framealpha=0.85)

        # Also save individual detector figure
        fig, ax = plt.subplots(figsize=(10.0, 5.0), dpi=180)
        make_like_colab(fig, ax)
        b1_i = ax.bar(x - w / 2, v_pose, width=w, label="LO-RANSAC [PoseLib]", color=c_pose, zorder=3)
        b2_i = ax.bar(x + w / 2, v_supe, width=w, label="SupeRANSAC", color=c_supe, zorder=3)
        for rect in b1_i:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for rect in b2_i:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#990000")
        ax.set_xticks(x)
        ax.set_xticklabels(detectors, fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, dyn_ymax)
        ax.set_ylabel(ylabel, fontsize=13.5, fontweight="bold")
        ax.set_title(title, fontsize=14.5, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=11.5, frameon=True, facecolor="white", framealpha=0.85)
        plt.tight_layout()
        out_p = os.path.join(ASSETS_DIR, fname)
        plt.savefig(out_p, dpi=180)
        plt.close(fig)
        print(f"Saved {out_p}")

    plt.close(fig_all)


def _eval_curve_point(args):
    det_key, prob_key, m_key, b, scenes, pairs_data = args
    errs = []
    dts = []
    mod = _get_mod(SUPERANSAC_SO) if m_key == "superansac" else None

    for s, (corrs, scores) in zip(scenes, pairs_data):
        min_n = 8 if prob_key in ("F", "E") else 4
        if len(corrs) < min_n:
            errs.append(180.0 if prob_key in ("F", "E") else 1e6)
            dts.append(0.001)
            continue

        img_sz = np.array(s["image_sizes"], dtype=np.float64)
        p1 = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
        p2 = np.ascontiguousarray(corrs[:, 2:4], dtype=np.float64)

        if m_key == "superansac":
            cfg = mod.RANSACSettings()
            cfg.min_iterations = b
            cfg.max_iterations = b
            if b < 1000:
                cfg.local_optimization_settings.max_iterations = max(5, min(50, b // 4))
                cfg.local_optimization_settings.graph_cut_number = max(2, min(20, b // 15))

            t0 = time.perf_counter()
            if prob_key == "F":
                F_est, inliers, _, _ = mod.estimateFundamentalMatrix(corrs, img_sz, scores, config=cfg)
                dt = time.perf_counter() - t0
                dts.append(dt)
                if F_est is None or np.all(F_est == 0) or np.any(np.isnan(F_est)) or len(inliers) < 7:
                    errs.append(180.0)
                    continue
                K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
                E_est = K2.T @ F_est @ K1
                inl_pts = corrs[inliers]
                p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
                p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
                _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
                errs.append(max(rotation_error_deg(R_gt, R_est), angular_translation_error_deg(t_gt, t_est)))
            elif prob_key == "E":
                K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
                E_est, inliers, _, _ = mod.estimateEssentialMatrix(corrs, K1, K2, img_sz, scores, config=cfg)
                dt = time.perf_counter() - t0
                dts.append(dt)
                if E_est is None or np.all(E_est == 0) or np.any(np.isnan(E_est)) or len(inliers) < 5:
                    errs.append(180.0)
                    continue
                inl_pts = corrs[inliers]
                p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
                p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
                _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
                errs.append(max(rotation_error_deg(R_gt, R_est), angular_translation_error_deg(t_gt, t_est)))
            else:
                H_gt, corners = s["H_gt"], s["corners"]
                H_est, inliers, _, _ = mod.estimateHomography(corrs, img_sz, scores, config=cfg)
                dt = time.perf_counter() - t0
                dts.append(dt)
                if H_est is None or np.all(H_est == 0) or np.any(np.isnan(H_est)) or len(inliers) < 4:
                    errs.append(1e6)
                    continue
                c_gt = corners @ H_gt.T
                c_gt = c_gt[:, :2] / (c_gt[:, 2:3] + 1e-12)
                c_est = corners @ H_est.T
                if np.any(np.abs(c_est[:, 2]) < 1e-8):
                    errs.append(1e6)
                    continue
                c_est = c_est[:, :2] / c_est[:, 2:3]
                errs.append(float(np.mean(np.linalg.norm(c_est - c_gt, axis=1))))

        elif m_key == "poselib":
            t0 = time.perf_counter()
            if prob_key == "F":
                F_est, info = poselib.estimate_fundamental(
                    p1, p2,
                    {"max_epipolar_error": 1.5, "progressive_sampling": True, "max_iterations": b, "min_iterations": b},
                    {"loss_type": "CAUCHY", "loss_scale": 0.75},
                )
                dt = time.perf_counter() - t0
                dts.append(dt)
                inl = np.asarray(info.get("inliers", []), dtype=bool)
                if F_est is None or np.sum(inl) < 7:
                    errs.append(180.0)
                    continue
                K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
                E_est = K2.T @ F_est @ K1
                p1_n = cv2.undistortPoints(p1[inl].reshape(-1, 1, 2), K1, None)
                p2_n = cv2.undistortPoints(p2[inl].reshape(-1, 1, 2), K2, None)
                _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
                errs.append(max(rotation_error_deg(R_gt, R_est), angular_translation_error_deg(t_gt, t_est)))
            elif prob_key == "E":
                K1, K2, R_gt, t_gt = s["K1"], s["K2"], s["R_gt"], s["t_gt"]
                cam1 = {"model": "PINHOLE", "width": int(img_sz[1]), "height": int(img_sz[0]), "params": [float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])]}
                cam2 = {"model": "PINHOLE", "width": int(img_sz[3]), "height": int(img_sz[2]), "params": [float(K2[0, 0]), float(K2[1, 1]), float(K2[0, 2]), float(K2[1, 2])]}
                pose, info = poselib.estimate_relative_pose(
                    p1, p2, cam1, cam2,
                    {"max_epipolar_error": 1.5, "progressive_sampling": True, "max_iterations": b, "min_iterations": b},
                    {"loss_type": "CAUCHY", "loss_scale": 0.75},
                )
                dt = time.perf_counter() - t0
                dts.append(dt)
                if pose is None or np.sum(info.get("inliers", [])) < 5:
                    errs.append(180.0)
                    continue
                errs.append(max(rotation_error_deg(R_gt, pose.R), angular_translation_error_deg(t_gt, pose.t)))
            else:
                H_gt, corners = s["H_gt"], s["corners"]
                H_est, info = poselib.estimate_homography(
                    p1, p2,
                    {"max_reproj_error": 4.0, "progressive_sampling": True, "max_iterations": b, "min_iterations": b},
                    {"loss_type": "CAUCHY", "loss_scale": 2.0},
                )
                dt = time.perf_counter() - t0
                dts.append(dt)
                if H_est is None or np.sum(info.get("inliers", [])) < 4:
                    errs.append(1e6)
                    continue
                c_gt = corners @ H_gt.T
                c_gt = c_gt[:, :2] / (c_gt[:, 2:3] + 1e-12)
                c_est = corners @ H_est.T
                if np.any(np.abs(c_est[:, 2]) < 1e-8):
                    errs.append(1e6)
                    continue
                c_est = c_est[:, :2] / c_est[:, 2:3]
                errs.append(float(np.mean(np.linalg.norm(c_est - c_gt, axis=1))))

    thresh = 5.0 if prob_key == "H" else 10.0
    return det_key, prob_key, m_key, b, compute_auc(errs, [thresh])[0], float(np.mean(dts))


def plot_speed_accuracy_curves():
    scenes_f, scenes_e, scenes_h = build_real_scenes()
    match_cache = get_or_extract_all_matches(scenes_f, scenes_e, scenes_h)

    scenes_by_prob = {"F": scenes_f, "E": scenes_e, "H": scenes_h}
    iter_budgets = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    methods = [
        ("superansac", "SupeRANSAC", mpl.colormaps["tab10"](3), "o", "-"),
        ("poselib", "LO-RANSAC [PoseLib]", mpl.colormaps["tab10"](2), "h", "-"),
    ]
    det_specs = [
        ("DISK+LG", "SP+LG", "splg"),
        ("RoMa", "RoMA", "roma"),
        ("LoMa", "LoMa", "loma"),
    ]
    prob_specs = [
        ("E", "essential_matrix", "AUC @ 10°"),
        ("F", "fundamental_matrix", "AUC @ 10°"),
        ("H", "homography_matrix", "AUC @ 5px"),
    ]

    tasks = []
    for det_key, _, _ in det_specs:
        for prob_key, _, _ in prob_specs:
            scenes = scenes_by_prob[prob_key]
            pairs_data = [match_cache[s["key"]][det_key] for s in scenes]
            for m_key, _, _, _, _ in methods:
                for b in iter_budgets:
                    tasks.append((det_key, prob_key, m_key, b, scenes, pairs_data))

    t0 = time.perf_counter()
    results = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=40, mp_context=ctx) as ex:
        for det_key, prob_key, m_key, b, auc_val, avg_s in ex.map(_eval_curve_point, tasks):
            results[(det_key, prob_key, m_key, b)] = (auc_val, avg_s)
    print(f"Evaluated {len(tasks)} curve points in {time.perf_counter() - t0:.2f}s")

    for det_key, det_title, det_suffix in det_specs:
        for prob_key, prob_prefix, ylabel in prob_specs:
            fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=180)
            make_like_colab(fig, ax)

            for m_key, m_label, color, marker, ls in methods:
                xs, ys = [], []
                for b in iter_budgets:
                    auc_val, avg_s = results[(det_key, prob_key, m_key, b)]
                    xs.append(max(avg_s, 1e-4))
                    ys.append(auc_val)
                lw = 3.0 if m_key == "superansac" else 2.5
                ms = 12 if m_key == "superansac" else 10
                z = 5 if m_key == "superansac" else 3
                ax.plot(xs, ys, label=m_label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms, zorder=z)

            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Average time [s]", fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_title(f"Speed-accuracy trade-off [{det_title}]", fontsize=22)
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.legend(loc="upper left", fontsize=14, frameon=True, facecolor="white", framealpha=0.85)
            plt.tight_layout()
            out_path = os.path.join(ASSETS_DIR, f"{prob_prefix}_{det_suffix}.png")
            plt.savefig(out_path, dpi=180)
            plt.close(fig)
            print(f"Saved {out_path}")


def plot_abspose_and_rigid_charts():
    """Evaluate and plot OpenCV RANSAC vs. SupeRANSAC on Absolute Pose (2D-3D) and Rigid Transformation (3D-3D)."""
    mod = _get_mod(SUPERANSAC_SO)
    c_cv = mpl.colormaps["tab10"](0)
    c_supe = mpl.colormaps["tab10"](3)

    # 1. Absolute Pose (2D-3D PnP on pose6dscene across outlier ratios 33% -> 91%)
    data_dir = os.path.join(ROOT, "superansac", "examples", "data")
    corrs_gt = np.loadtxt(os.path.join(data_dir, "pose6dscene_points.txt"))
    gt_pose = np.loadtxt(os.path.join(data_dir, "pose6dscene_gt.txt"))
    K = np.loadtxt(os.path.join(data_dir, "pose6dscene.K"))
    R_gt_pnp = gt_pose[:, :3]
    t_gt_pnp = gt_pose[:, 3]
    cam_params = [float(K[0, 0]), float(K[0, 2]), float(K[1, 2])]
    inv_K = np.linalg.inv(K)

    outlier_mults = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    outlier_labels = ["33%", "50%", "67%", "75%", "83%", "88%", "91%"]
    n_trials = 20

    cv_all_rot, cv_all_trans, cv_all_comb = [], [], []
    supe_all_rot, supe_all_trans, supe_all_comb = [], [], []
    cv_by_mult_auc10, supe_by_mult_auc10 = [], []
    cv_by_mult_rot, supe_by_mult_rot = [], []
    cv_by_mult_trans, supe_by_mult_trans = [], []

    min_gt = corrs_gt.min(axis=0)
    max_gt = corrs_gt.max(axis=0)

    for m_idx, mult in enumerate(outlier_mults):
        rng = np.random.default_rng(2026 + m_idx * 37)
        n_out = int(round(mult * len(corrs_gt)))
        cv_r_m, cv_t_m, cv_c_m = [], [], []
        su_r_m, su_t_m, su_c_m = [], [], []

        for _ in range(n_trials):
            outliers = rng.uniform(min_gt, max_gt, size=(n_out, 5))
            pts = np.vstack([corrs_gt, outliers])
            pts[:, :2] += rng.normal(0.0, 1.5, size=(len(pts), 2))
            pts = np.ascontiguousarray(pts[rng.permutation(len(pts))], dtype=np.float64)

            # OpenCV RANSAC baseline (cv2.solvePnPRansac from example_absolute_pose_fitting.ipynb)
            p1_h = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
            p2_n = (inv_K @ p1_h.T).T[:, :2]
            image_pts = np.float32(p2_n).reshape(-1, 2)
            world_pts = np.float32(pts[:, 2:5]).reshape(-1, 3)
            norm_thresh = 3.5 / (K[0, 0] + K[1, 1]) / 2.0
            ok, rvec, tvec, inl_cv = cv2.solvePnPRansac(
                world_pts,
                image_pts,
                np.identity(3),
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=5000,
                reprojectionError=norm_thresh,
            )
            if ok and rvec is not None and tvec is not None and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
                R_cv, _ = cv2.Rodrigues(rvec)
                t_cv = tvec.flatten()
                er_cv = rotation_error_deg(R_gt_pnp, R_cv)
                et_cv = float(np.linalg.norm(t_gt_pnp - t_cv))  # scene units are already cm
            else:
                er_cv, et_cv = 180.0, 1000.0

            # SupeRANSAC (from example_absolute_pose_fitting.ipynb)
            pts2d = pts[:, :2]
            pts3d = pts[:, 2:5]
            min_c = np.min(pts3d, axis=0)
            pts3d_shifted = pts3d - min_c
            corrs_su = np.ascontiguousarray(np.hstack([pts2d, pts3d_shifted]), dtype=np.float64)
            bbox = list(np.max(corrs_su, axis=0))

            cfg = mod.RANSACSettings()
            cfg.inlier_threshold = 3.5
            R_su, t_su_raw, inl_su, _, _ = mod.estimateAbsolutePose(
                corrs_su, mod.CameraType.SimplePinhole, cam_params, bbox, [], config=cfg
            )
            if R_su is not None and len(inl_su) >= 4:
                t_su = t_su_raw.flatten() - R_su @ min_c
                er_su = rotation_error_deg(R_gt_pnp, R_su)
                et_su = float(np.linalg.norm(t_gt_pnp - t_su))  # scene units are already cm
            else:
                er_su, et_su = 180.0, 1000.0

            comb_cv = max(er_cv, et_cv * (10.0 / 50.0))  # normalized to (10 deg, 50 cm)
            comb_su = max(er_su, et_su * (10.0 / 50.0))

            cv_r_m.append(er_cv)
            cv_t_m.append(et_cv)
            cv_c_m.append(comb_cv)
            su_r_m.append(er_su)
            su_t_m.append(et_su)
            su_c_m.append(comb_su)

        cv_all_rot.extend(cv_r_m)
        cv_all_trans.extend(cv_t_m)
        cv_all_comb.extend(cv_c_m)
        supe_all_rot.extend(su_r_m)
        supe_all_trans.extend(su_t_m)
        supe_all_comb.extend(su_c_m)

        cv_by_mult_auc10.append(compute_auc(cv_c_m, [10.0])[0])
        supe_by_mult_auc10.append(compute_auc(su_c_m, [10.0])[0])
        cv_by_mult_rot.append(float(np.median(cv_r_m)))
        supe_by_mult_rot.append(float(np.median(su_r_m)))
        cv_by_mult_trans.append(float(np.median(cv_t_m)))
        supe_by_mult_trans.append(float(np.median(su_t_m)))

    pnp_cv_auc5 = compute_auc(cv_all_comb, [5.0])[0]
    pnp_cv_auc10 = compute_auc(cv_all_comb, [10.0])[0]
    pnp_cv_auc20 = compute_auc(cv_all_comb, [20.0])[0]
    pnp_su_auc5 = compute_auc(supe_all_comb, [5.0])[0]
    pnp_su_auc10 = compute_auc(supe_all_comb, [10.0])[0]
    pnp_su_auc20 = compute_auc(supe_all_comb, [20.0])[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    make_like_colab(fig, ax1)
    make_like_colab(fig, ax2)

    m_labels = ["AUC@(5°,25cm)", "AUC@(10°,50cm)", "AUC@(20°,1m)"]
    cv_vals = [pnp_cv_auc5, pnp_cv_auc10, pnp_cv_auc20]
    su_vals = [pnp_su_auc5, pnp_su_auc10, pnp_su_auc20]
    x = np.arange(len(m_labels))
    w = 0.34
    b1 = ax1.bar(x - w / 2, cv_vals, width=w, label="OpenCV RANSAC", color=c_cv, zorder=3)
    b2 = ax1.bar(x + w / 2, su_vals, width=w, label="SupeRANSAC", color=c_supe, zorder=3)
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#990000")
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_labels, fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Pose AUC Score", fontsize=12.5, fontweight="bold")
    ax1.set_title("6D Absolute Pose (PnP): Overall AUC", fontsize=13.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=10.5, frameon=True, facecolor="white", framealpha=0.85)

    xo = np.arange(len(outlier_labels))
    ax2.plot(xo, supe_by_mult_auc10, label="SupeRANSAC", color=c_supe, marker="o", linewidth=2.8, markersize=8, zorder=5)
    ax2.plot(xo, cv_by_mult_auc10, label="OpenCV RANSAC", color=c_cv, marker="s", linewidth=2.4, markersize=7, zorder=4)
    ax2.set_xticks(xo)
    ax2.set_xticklabels(outlier_labels, fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Outlier Ratio", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("Pose AUC @ (10°, 50cm)", fontsize=12.5, fontweight="bold")
    ax2.set_title("PnP Robustness vs. Outlier Ratio", fontsize=13.5, fontweight="bold", pad=10)
    ax2.legend(loc="lower left", fontsize=10.5, frameon=True, facecolor="white", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "absolute_pose_experiments.png"), dpi=180)
    plt.close(fig)
    print("Saved absolute_pose_experiments.png")

    # 2. 6D Rigid Transformation (3D-3D on benchmark_data.pkl, 100 pairs across 4 difficulty tiers)
    with open(BENCHMARK_DATA_PATH, "rb") as f:
        bdata = pickle.load(f)
    rigid_pairs = bdata["rigid_pose"]

    cv_r_all, cv_t_all, cv_c_all = [], [], []
    su_r_all, su_t_all, su_c_all = [], [], []
    tier_names = ["Tier 0", "Tier 1", "Tier 2", "Tier 3"]
    cv_tier_c = {i: [] for i in range(4)}
    su_tier_c = {i: [] for i in range(4)}

    for idx, item in enumerate(rigid_pairs):
        corrs = np.ascontiguousarray(item["corrs"], dtype=np.float64)
        probs = np.ascontiguousarray(item["scores"], dtype=np.float64)
        T_gt = item["T_gt"]
        th = item["threshold"]
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        tier = min(3, idx // (max(1, len(rigid_pairs) // 4)))

        # OpenCV RANSAC baseline (cv2.estimateAffine3D + SVD orthogonalization at matched 0.275*th threshold)
        src = corrs[:, :3].astype(np.float64)
        dst = corrs[:, 3:6].astype(np.float64)
        retval, M_cv, inl_cv = cv2.estimateAffine3D(src, dst, ransacThreshold=0.275 * th, confidence=0.9999)
        if retval and M_cv is not None and inl_cv is not None and np.sum(inl_cv) >= 3:
            mask = inl_cv.ravel() == 1
            s_inl, d_inl = src[mask], dst[mask]
            c_s, c_d = s_inl.mean(axis=0), d_inl.mean(axis=0)
            H_cov = (s_inl - c_s).T @ (d_inl - c_d)
            U, _, Vt = np.linalg.svd(H_cov)
            R_cv = Vt.T @ U.T
            if np.linalg.det(R_cv) < 0:
                Vt[-1, :] *= -1
                R_cv = Vt.T @ U.T
            t_cv = c_d - R_cv @ c_s
            er_cv = rotation_error_deg(R_gt, R_cv)
            et_cv = float(np.linalg.norm(t_gt - t_cv) * 100.0)
        else:
            er_cv, et_cv = 180.0, 1000.0

        # SupeRANSAC
        min_coords = np.min(corrs, axis=0)
        T1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-min_coords[0], -min_coords[1], -min_coords[2], 1],
        ], dtype=np.float64)
        T2inv = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [min_coords[3], min_coords[4], min_coords[5], 1],
        ], dtype=np.float64)
        shifted = np.ascontiguousarray(corrs - min_coords, dtype=np.float64)
        bbox = np.ascontiguousarray(np.max(shifted, axis=0) + 1e-3, dtype=np.float64)
        cfg = mod.RANSACSettings()
        cfg.inlier_threshold = th
        T_est_raw, inl_su, _, _ = mod.estimateRigidTransform(shifted, bbox, probs, config=cfg)
        if T_est_raw is not None and len(inl_su) >= 3:
            T_su_T = (T1 @ T_est_raw @ T2inv).T
            R_su = T_su_T[:3, :3]
            t_su = T_su_T[:3, 3]
            er_su = rotation_error_deg(R_gt, R_su)
            et_su = float(np.linalg.norm(t_gt - t_su) * 100.0)
        else:
            er_su, et_su = 180.0, 1000.0

        comb_cv = max(er_cv, et_cv * 0.5)
        comb_su = max(er_su, et_su * 0.5)
        cv_r_all.append(er_cv)
        cv_t_all.append(et_cv)
        cv_c_all.append(comb_cv)
        su_r_all.append(er_su)
        su_t_all.append(et_su)
        su_c_all.append(comb_su)
        cv_tier_c[tier].append(comb_cv)
        su_tier_c[tier].append(comb_su)

    rig_cv_pose = compute_auc(cv_c_all, [5.0])[0]
    rig_cv_rot = compute_auc(cv_r_all, [5.0])[0]
    rig_cv_trans = compute_auc(cv_t_all, [10.0])[0]
    rig_su_pose = compute_auc(su_c_all, [5.0])[0]
    rig_su_rot = compute_auc(su_r_all, [5.0])[0]
    rig_su_trans = compute_auc(su_t_all, [10.0])[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    make_like_colab(fig, ax1)
    make_like_colab(fig, ax2)

    r_labels = ["Pose AUC\n@(5°,10cm)", "Rotation AUC\n@5°", "Translation AUC\n@10cm"]
    cv_rvals = [rig_cv_pose, rig_cv_rot, rig_cv_trans]
    su_rvals = [rig_su_pose, rig_su_rot, rig_su_trans]
    xr = np.arange(len(r_labels))
    w = 0.34
    b1 = ax1.bar(xr - w / 2, cv_rvals, width=w, label="OpenCV RANSAC", color=c_cv, zorder=3)
    b2 = ax1.bar(xr + w / 2, su_rvals, width=w, label="SupeRANSAC", color=c_supe, zorder=3)
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#990000")
    ax1.set_xticks(xr)
    ax1.set_xticklabels(r_labels, fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 1.15)
    ax1.set_ylabel("AUC Score", fontsize=12.5, fontweight="bold")
    ax1.set_title("6D Rigid Registration (SE(3)): Overall AUC", fontsize=13.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=10.5, frameon=True, facecolor="white", framealpha=0.85)

    xt = np.arange(4)
    t_labels = ["Tier 0\n(Clean)", "Tier 1\n(Moderate)", "Tier 2\n(High Outl.)", "Tier 3\n(Extreme)"]
    cv_tvals = [compute_auc(cv_tier_c[i], [5.0])[0] for i in range(4)]
    su_tvals = [compute_auc(su_tier_c[i], [5.0])[0] for i in range(4)]
    bt1 = ax2.bar(xt - w / 2, cv_tvals, width=w, label="OpenCV RANSAC", color=c_cv, zorder=3)
    bt2 = ax2.bar(xt + w / 2, su_tvals, width=w, label="SupeRANSAC", color=c_supe, zorder=3)
    for rect in bt1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    for rect in bt2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2.0, h + 0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#990000")
    ax2.set_xticks(xt)
    ax2.set_xticklabels(t_labels, fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0.0, 1.15)
    ax2.set_ylabel("Pose AUC @ (5°, 10cm)", fontsize=12.5, fontweight="bold")
    ax2.set_title("SE(3) Registration Across Difficulty Tiers", fontsize=13.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=10.5, frameon=True, facecolor="white", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "rigid_transform_experiments.png"), dpi=180)
    plt.close(fig)
    print("Saved rigid_transform_experiments.png")

    summary = {
        "absolute_pose": {
            "opencv": {
                "auc_5deg_25cm": pnp_cv_auc5,
                "auc_10deg_50cm": pnp_cv_auc10,
                "auc_20deg_100cm": pnp_cv_auc20,
                "median_rot_deg": float(np.median(cv_all_rot)),
                "median_trans_cm": float(np.median(cv_all_trans)),
                "by_outlier_ratio_auc10": dict(zip(outlier_labels, cv_by_mult_auc10)),
                "by_outlier_ratio_rot": dict(zip(outlier_labels, cv_by_mult_rot)),
                "by_outlier_ratio_trans": dict(zip(outlier_labels, cv_by_mult_trans)),
            },
            "superansac": {
                "auc_5deg_25cm": pnp_su_auc5,
                "auc_10deg_50cm": pnp_su_auc10,
                "auc_20deg_100cm": pnp_su_auc20,
                "median_rot_deg": float(np.median(supe_all_rot)),
                "median_trans_cm": float(np.median(supe_all_trans)),
                "by_outlier_ratio_auc10": dict(zip(outlier_labels, supe_by_mult_auc10)),
                "by_outlier_ratio_rot": dict(zip(outlier_labels, supe_by_mult_rot)),
                "by_outlier_ratio_trans": dict(zip(outlier_labels, supe_by_mult_trans)),
            },
        },
        "rigid_pose": {
            "opencv": {
                "pose_auc_5deg_10cm": rig_cv_pose,
                "rot_auc_5deg": rig_cv_rot,
                "trans_auc_10cm": rig_cv_trans,
                "by_tier_auc": cv_tvals,
            },
            "superansac": {
                "pose_auc_5deg_10cm": rig_su_pose,
                "rot_auc_5deg": rig_su_rot,
                "trans_auc_10cm": rig_su_trans,
                "by_tier_auc": su_tvals,
            },
        },
    }
    with open(os.path.join(SCRIPT_DIR, "abspose_rigid_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved abspose_rigid_results.json")

    print("\n=== Section 4.1: 6D Absolute Pose (PnP) — OpenCV RANSAC vs. SupeRANSAC ===")
    print(f"Overall AUC@(5°,25cm):  OpenCV = {pnp_cv_auc5:.4f} | SupeRANSAC = {pnp_su_auc5:.4f} ({(pnp_su_auc5-pnp_cv_auc5)*100:+.2f}pt)")
    print(f"Overall AUC@(10°,50cm): OpenCV = {pnp_cv_auc10:.4f} | SupeRANSAC = {pnp_su_auc10:.4f} ({(pnp_su_auc10-pnp_cv_auc10)*100:+.2f}pt)")
    print(f"Overall AUC@(20°,1m):   OpenCV = {pnp_cv_auc20:.4f} | SupeRANSAC = {pnp_su_auc20:.4f} ({(pnp_su_auc20-pnp_cv_auc20)*100:+.2f}pt)")
    print(f"Median Rotation Error:  OpenCV = {np.median(cv_all_rot):.2f}° | SupeRANSAC = {np.median(supe_all_rot):.2f}°")
    print(f"Median Trans. Error:    OpenCV = {np.median(cv_all_trans):.2f} cm | SupeRANSAC = {np.median(supe_all_trans):.2f} cm")

    print("\n=== Section 4.2: 6D Rigid Registration (SE(3)) — OpenCV RANSAC vs. SupeRANSAC ===")
    print(f"Overall Pose AUC@(5°,10cm): OpenCV = {rig_cv_pose:.4f} | SupeRANSAC = {rig_su_pose:.4f} ({(rig_su_pose-rig_cv_pose)*100:+.2f}pt)")
    print(f"Overall Rotation AUC@5°:    OpenCV = {rig_cv_rot:.4f} | SupeRANSAC = {rig_su_rot:.4f} ({(rig_su_rot-rig_cv_rot)*100:+.2f}pt)")
    print(f"Overall Trans. AUC@10cm:    OpenCV = {rig_cv_trans:.4f} | SupeRANSAC = {rig_su_trans:.4f} ({(rig_su_trans-rig_cv_trans)*100:+.2f}pt)")
    for i, tname in enumerate(tier_names):
        print(f"  {tname}: OpenCV = {cv_tvals[i]:.4f} | SupeRANSAC = {su_tvals[i]:.4f} ({(su_tvals[i]-cv_tvals[i])*100:+.2f}pt)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate README plots and tables for Sections 1-4")
    parser.add_argument(
        "--section",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Which README section to reproduce: 1 (Overview), 2 (Speed-Accuracy Curves), 3 (7-Matcher Evaluation), 4 (Absolute & Rigid Pose), or all.",
    )
    args = parser.parse_args()
    if args.section in ("4", "all"):
        plot_abspose_and_rigid_charts()
    if args.section in ("1", "all"):
        plot_section1_overview()
    if args.section in ("2", "all"):
        plot_speed_accuracy_curves()
    if args.section in ("3", "all"):
        plot_section3_detectors()
