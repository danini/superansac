"""Runnable benchmark script for 6D Absolute Pose (PnP) and 6D Rigid Registration (SE(3)).

Reproduces:
- Experiment 4a: 6D Absolute Pose (PnP) — SupeRANSAC vs. PoseLib & OpenCV RANSAC
  across 7Scenes RGB-D (2D-3D) and pose6dscene (assets/absolute_pose_experiments.png)
- Experiment 4b: 6D Rigid Registration (SE(3)) — SupeRANSAC vs. OpenCV RANSAC
  across 7Scenes RGB-D (3D-3D) and rigid_pose_example (assets/rigid_transform_experiments.png)
- Multi-Dataset Overview Figure (assets/benchmark_overview.png)

Usage:
    python tests/test_6d_pose.py --problem all --dataset all
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERANSAC_DIR = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import poselib
import pysuperansac

from datasets.multi_benchmark import (
    DEFAULT_DATA_DIR,
    load_7scenes_pnp_problems,
    load_7scenes_rigid_problems,
    load_pose6dscene_pnp_problems,
    load_rigid_pose_example_problems,
)
from evaluation import pose_auc, rotation_error_from_trace

cv2.setNumThreads(1)

BUDGETS = [25, 50, 100, 250, 500, 1000, 2500, 5000]


def _eval_pnp_problem(args):
    prob, max_iters, min_iters = args
    corrs = prob["corrs"]
    probs = prob["probs"]
    K = prob["K"]
    R_gt = prob["R_gt"]
    t_gt = prob["t_gt"]
    th = prob["threshold_px"]
    cm_scale = prob["cm_scale"]
    comb_w = prob["comb_trans_weight"]

    pts2d = np.ascontiguousarray(corrs[:, 0:2], dtype=np.float64)
    pts3d = np.ascontiguousarray(corrs[:, 2:5], dtype=np.float64)
    min_c = np.min(pts3d, axis=0)
    pts3d_shifted = pts3d - min_c
    corrs_su = np.ascontiguousarray(np.hstack([pts2d, pts3d_shifted]), dtype=np.float64)
    bbox = list(np.max(corrs_su, axis=0))
    cam_params_su = [float(K[0, 0]), float(K[0, 2]), float(K[1, 2])]

    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold = th
    cfg.min_iterations = min_iters
    cfg.max_iterations = max_iters
    if max_iters < 1000:
        cfg.local_optimization_settings.max_iterations = max(5, min(50, max_iters // 4))
        cfg.local_optimization_settings.graph_cut_number = max(2, min(20, max_iters // 15))

    t0 = time.perf_counter()
    R_su, t_su_raw, inl_su, _, _ = pysuperansac.estimateAbsolutePose(
        corrs_su,
        pysuperansac.CameraType.SimplePinhole,
        cam_params_su,
        bbox,
        probs if len(probs) > 0 else [],
        config=cfg,
    )
    t_su_ms = (time.perf_counter() - t0) * 1000.0
    if R_su is None or len(inl_su) < 4 or not np.all(np.isfinite(R_su)) or not np.all(np.isfinite(t_su_raw)):
        r_su, tr_su, comb_su = 180.0, 1e6, 180.0
    else:
        t_su = np.asarray(t_su_raw, dtype=np.float64).flatten() - R_su @ min_c
        r_su = float(rotation_error_from_trace(R_gt, R_su))
        tr_su = float(np.linalg.norm(t_gt - t_su) * cm_scale)
        comb_su = float(max(r_su, tr_su * comb_w))

    t0 = time.perf_counter()
    ok_cv, rvec_cv, tvec_cv, inl_cv_arr = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K.astype(np.float64),
        None,
        iterationsCount=max_iters,
        reprojectionError=float(th),
        confidence=0.9999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    t_cv_ms = (time.perf_counter() - t0) * 1000.0
    inl_cv = 0 if (not ok_cv or inl_cv_arr is None) else len(inl_cv_arr)
    if not ok_cv or inl_cv < 4 or rvec_cv is None or tvec_cv is None or not np.all(np.isfinite(rvec_cv)) or not np.all(np.isfinite(tvec_cv)):
        r_cv, tr_cv, comb_cv = 180.0, 1e6, 180.0
    else:
        R_cv, _ = cv2.Rodrigues(rvec_cv)
        t_cv = tvec_cv.flatten()
        r_cv = float(rotation_error_from_trace(R_gt, R_cv))
        tr_cv = float(np.linalg.norm(t_gt - t_cv) * cm_scale)
        comb_cv = float(max(r_cv, tr_cv * comb_w))

    return {
        "dataset": prob["dataset"],
        "detector": prob.get("detector", "All"),
        "su": (comb_su, r_su, tr_su, t_su_ms, len(inl_su)),
        "cv": (comb_cv, r_cv, tr_cv, t_cv_ms, inl_cv),
    }


def _eval_rigid_problem(args):
    prob, max_iters, min_iters = args
    corrs = prob["corrs"]
    probs = prob["probs"]
    R_gt = prob["R_gt"]
    t_gt = prob["t_gt"]
    th = prob["threshold_3d"]
    cm_scale = prob["cm_scale"]
    comb_w = prob["comb_trans_weight"]

    min_coords = np.min(corrs, axis=0)
    T1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-min_coords[0], -min_coords[1], -min_coords[2], 1.0],
    ], dtype=np.float64)
    T2inv = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [min_coords[3], min_coords[4], min_coords[5], 1.0],
    ], dtype=np.float64)
    shifted = np.ascontiguousarray(corrs - min_coords, dtype=np.float64)
    bbox = np.ascontiguousarray(np.max(shifted, axis=0) + 1e-3, dtype=np.float64)

    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold = th
    cfg.min_iterations = min_iters
    cfg.max_iterations = max_iters
    if max_iters < 1000:
        cfg.local_optimization_settings.max_iterations = max(5, min(50, max_iters // 4))
        cfg.local_optimization_settings.graph_cut_number = max(2, min(20, max_iters // 15))

    t0 = time.perf_counter()
    T_est_raw, inl_su, _, _ = pysuperansac.estimateRigidTransform(
        shifted, bbox, probs, config=cfg
    )
    t_su_ms = (time.perf_counter() - t0) * 1000.0
    if T_est_raw is None or len(inl_su) < 3 or not np.all(np.isfinite(T_est_raw)):
        r_su, tr_su, comb_su = 180.0, 1e6, 180.0
    else:
        T_su_T = (T1 @ T_est_raw @ T2inv).T
        R_est = T_su_T[:3, :3]
        t_est = T_su_T[:3, 3].flatten()
        r_su = float(rotation_error_from_trace(R_gt, R_est))
        tr_su = float(np.linalg.norm(t_gt - t_est) * cm_scale)
        comb_su = float(max(r_su, tr_su * comb_w))

    src3d = np.ascontiguousarray(corrs[:, :3], dtype=np.float64)
    dst3d = np.ascontiguousarray(corrs[:, 3:6], dtype=np.float64)
    t0 = time.perf_counter()
    ok_cv, M_cv, inl_cv_arr = cv2.estimateAffine3D(
        src3d, dst3d, ransacThreshold=0.275 * th, confidence=0.9999
    )
    t_cv_ms = (time.perf_counter() - t0) * 1000.0
    inl_cv = 0 if (not ok_cv or inl_cv_arr is None) else int(np.sum(inl_cv_arr))
    if not ok_cv or M_cv is None or inl_cv < 3 or not np.all(np.isfinite(M_cv)):
        r_cv, tr_cv, comb_cv = 180.0, 1e6, 180.0
    else:
        mask = inl_cv_arr.ravel() == 1
        s_inl, d_inl = src3d[mask], dst3d[mask]
        c_s, c_d = s_inl.mean(axis=0), d_inl.mean(axis=0)
        H_cov = (s_inl - c_s).T @ (d_inl - c_d)
        U, _, Vt = np.linalg.svd(H_cov)
        R_cv = Vt.T @ U.T
        if np.linalg.det(R_cv) < 0:
            Vt[-1, :] *= -1
            R_cv = Vt.T @ U.T
        t_cv = c_d - R_cv @ c_s
        r_cv = float(rotation_error_from_trace(R_gt, R_cv))
        tr_cv = float(np.linalg.norm(t_gt - t_cv) * cm_scale)
        comb_cv = float(max(r_cv, tr_cv * comb_w))

    return {
        "dataset": prob["dataset"],
        "detector": prob.get("detector", "All"),
        "su": (comb_su, r_su, tr_su, t_su_ms, len(inl_su)),
        "cv": (comb_cv, r_cv, tr_cv, t_cv_ms, inl_cv),
    }


def summarize_6d_results(results, methods, auc_thresholds=(2.0, 5.0, 10.0)):
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)
    by_ds["Overall"] = results

    summary = {}
    for ds_name, ds_res in by_ds.items():
        summary[ds_name] = {"num_problems": len(ds_res)}
        for m_key, m_label in methods:
            pose_e = [r[m_key][0] for r in ds_res]
            rot_e = [r[m_key][1] for r in ds_res]
            tr_e = [r[m_key][2] for r in ds_res]
            ts = [r[m_key][3] for r in ds_res]
            inls = [r[m_key][4] for r in ds_res]
            aucs = [x * 100.0 for x in pose_auc(pose_e, auc_thresholds)]
            summary[ds_name][m_label] = {
                "auc": aucs,
                "med_rot": float(np.median(rot_e)),
                "med_trans": float(np.median(tr_e)),
                "time_ms": float(np.mean(ts)),
                "med_ms": float(np.median(ts)),
                "inliers": float(np.mean(inls)),
            }
    return summary


def plot_pnp_figure(summary, curves, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel 1: Overall AUC bar chart
    ax0 = axes[0]
    labels = ["AUC@(5°,25cm)", "AUC@(10°,50cm)", "AUC@(20°,1m)"]
    x = np.arange(len(labels))
    w = 0.35
    ov = summary["Overall"]
    su_v = ov["SupeRANSAC"]["auc"]
    cv_v = ov["OpenCV RANSAC"]["auc"]

    b1 = ax0.bar(x - w / 2, su_v, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax0.bar(x + w / 2, cv_v, w, label="OpenCV RANSAC", color="#e41a1c", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax0.set_ylabel("Pose AUC (%)", fontsize=10.5, fontweight="bold")
    ax0.set_title("6D Absolute Pose (PnP) — Overall Accuracy", fontsize=11, fontweight="bold")
    ax0.set_ylim(0, max(max(su_v), max(cv_v)) * 1.18 + 4.0)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(loc="upper left", frameon=True, fontsize=9.0)

    # Panel 2: Per-dataset AUC@(10°,50cm) comparison
    ax1 = axes[1]
    ds_names = [k for k in summary.keys() if k != "Overall"] + ["Overall"]
    xd = np.arange(len(ds_names))
    su_d = [summary[d]["SupeRANSAC"]["auc"][1] for d in ds_names]
    cv_d = [summary[d]["OpenCV RANSAC"]["auc"][1] for d in ds_names]
    b1 = ax1.bar(xd - w / 2, su_d, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(xd + w / 2, cv_d, w, label="OpenCV RANSAC", color="#e41a1c", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_xticks(xd)
    ax1.set_xticklabels(ds_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("AUC@(10°,50cm) (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("PnP Accuracy by Dataset (AUC@10°/50cm)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(max(su_d), max(cv_d)) * 1.18 + 4.0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 3: Speed-Accuracy Trade-Off Curve
    ax2 = axes[2]
    for m_label, color, style in [
        ("SupeRANSAC", "#1b9e77", "-o"),
        ("OpenCV RANSAC", "#e41a1c", "--s"),
    ]:
        pts = curves[m_label]
        ax2.plot(
            [p["time_ms"] for p in pts],
            [p["auc"][1] for p in pts],
            style, color=color, linewidth=2.4, markersize=6.0,
            label="SupeRANSAC (Ours)" if m_label == "SupeRANSAC" else m_label,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Average Runtime (ms, log scale)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("AUC@(10°,50cm) (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Speed–Accuracy Trade-Off (25–5000 iters)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.0)

    fig.suptitle(
        "6D Absolute Pose (PnP) Evaluation — 7Scenes RGB-D & pose6dscene",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(assets_dir, "absolute_pose_experiments.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_rigid_figure(summary, results, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel 1: Overall AUC bar chart
    ax0 = axes[0]
    labels = ["AUC@(2.5°,5cm)", "AUC@(5°,10cm)", "AUC@(10°,20cm)"]
    x = np.arange(len(labels))
    w = 0.35
    ov = summary["Overall"]
    su_v = ov["SupeRANSAC"]["auc"]
    cv_v = ov["OpenCV RANSAC"]["auc"]

    b1 = ax0.bar(x - w / 2, su_v, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax0.bar(x + w / 2, cv_v, w, label="OpenCV RANSAC", color="#e41a1c", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax0.set_ylabel("Pose AUC (%)", fontsize=10.5, fontweight="bold")
    ax0.set_title("6D Rigid Registration — Overall Accuracy", fontsize=11, fontweight="bold")
    ax0.set_ylim(0, max(max(su_v), max(cv_v)) * 1.18 + 4.0)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(loc="upper left", frameon=True, fontsize=9.0)

    # Panel 2: Per-dataset AUC@(5°,10cm) comparison
    ax1 = axes[1]
    ds_names = [k for k in summary.keys() if k != "Overall"] + ["Overall"]
    xd = np.arange(len(ds_names))
    su_d = [summary[d]["SupeRANSAC"]["auc"][1] for d in ds_names]
    cv_d = [summary[d]["OpenCV RANSAC"]["auc"][1] for d in ds_names]
    b1 = ax1.bar(xd - w / 2, su_d, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(xd + w / 2, cv_d, w, label="OpenCV RANSAC", color="#e41a1c", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_xticks(xd)
    ax1.set_xticklabels(ds_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("AUC@(5°,10cm) (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("3D–3D Registration by Dataset (AUC@5°/10cm)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(max(su_d), max(cv_d)) * 1.18 + 4.0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 3: Accuracy across difficulty regimes
    ax2 = axes[2]
    regimes = [
        ("7Scenes\nRGB-D", lambda r: r["dataset"] == "7Scenes RGB-D"),
        ("Tier 1\n(Scan)", lambda r: r["detector"] == "tier1_real_scan"),
        ("Tier 2\n(Outlier)", lambda r: r["detector"] == "tier2_high_outlier"),
        ("Tier 3\n(Noise)", lambda r: r["detector"] == "tier3_sensor_noise"),
        ("Tier 4\n(Extreme)", lambda r: r["detector"] == "tier4_extreme"),
    ]
    reg_labels, su_r, cv_r = [], [], []
    for r_lbl, pred in regimes:
        sub = [r for r in results if pred(r)]
        if sub:
            reg_labels.append(r_lbl)
            su_r.append(pose_auc([r["su"][0] for r in sub], [5.0])[0] * 100.0)
            cv_r.append(pose_auc([r["cv"][0] for r in sub], [5.0])[0] * 100.0)
    xr = np.arange(len(reg_labels))
    b1 = ax2.bar(xr - w / 2, su_r, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax2.bar(xr + w / 2, cv_r, w, label="OpenCV RANSAC", color="#e41a1c", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(xr)
    ax2.set_xticklabels(reg_labels, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("AUC@(5°,10cm) (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Accuracy Across Difficulty Regimes", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(max(su_r), max(cv_r)) * 1.18 + 4.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "6D Rigid Transformation (3D–3D Registration) — 7Scenes RGB-D & Point Cloud Benchmark",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(assets_dir, "rigid_transform_experiments.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_benchmark_overview(data_dir, pnp_summary, rigid_summary, assets_dir):
    rel_path = os.path.join(data_dir, "relative_pose_results.json")
    hom_path = os.path.join(data_dir, "homography_results.json")
    if not os.path.exists(rel_path) or not os.path.exists(hom_path):
        return
    with open(rel_path, "r") as f:
        rel = json.load(f)
    with open(hom_path, "r") as f:
        hom = json.load(f)

    # Average across the 3 main matchers (LoMa, RoMa, DISK+LG) for F, E, H
    def _avg_feh(summary_dict):
        su_a, pl_a, su_t, pl_t = [], [], [], []
        for k in ("LoMa", "RoMa", "DISK+LG"):
            if k in summary_dict and "Overall" in summary_dict[k]:
                su_a.append(summary_dict[k]["Overall"]["SupeRANSAC"]["auc"])
                pl_a.append(summary_dict[k]["Overall"]["PoseLib"]["auc"])
                su_t.append(summary_dict[k]["Overall"]["SupeRANSAC"]["time_ms"])
                pl_t.append(summary_dict[k]["Overall"]["PoseLib"]["time_ms"])
        return (
            np.mean(su_a, axis=0).tolist(),
            np.mean(pl_a, axis=0).tolist(),
            float(np.mean(su_t)),
            float(np.mean(pl_t)),
        )

    f_su, f_pl, ft_su, ft_pl = _avg_feh(rel["fundamental"]["summary"])
    e_su, e_pl, et_su, et_pl = _avg_feh(rel["essential"]["summary"])
    h_su, h_pl, ht_su, ht_pl = _avg_feh(hom["summary"])
    p_su = pnp_summary["Overall"]["SupeRANSAC"]["auc"]
    p_cv = pnp_summary["Overall"]["OpenCV RANSAC"]["auc"]
    pt_su = pnp_summary["Overall"]["SupeRANSAC"]["time_ms"]
    pt_cv = pnp_summary["Overall"]["OpenCV RANSAC"]["time_ms"]
    r_su = rigid_summary["Overall"]["SupeRANSAC"]["auc"]
    r_cv = rigid_summary["Overall"]["OpenCV RANSAC"]["auc"]
    rt_su = rigid_summary["Overall"]["SupeRANSAC"]["time_ms"]
    rt_cv = rigid_summary["Overall"]["OpenCV RANSAC"]["time_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    tasks = ["Fundamental\n(AUC@10°)", "Essential\n(AUC@10°)", "Homography\n(AUC@5px)", "6D Abs Pose\n(AUC@10°/50cm)", "6D Rigid\n(AUC@5°/10cm)"]
    su_scores = [f_su[1], e_su[1], h_su[1], p_su[1], r_su[1]]
    base_scores = [f_pl[1], e_pl[1], h_pl[1], p_cv[1], r_cv[1]]
    su_times = [ft_su, et_su, ht_su, pt_su, rt_su]
    base_times = [ft_pl, et_pl, ht_pl, pt_cv, rt_cv]

    x = np.arange(len(tasks))
    w = 0.36
    ax0 = axes[0]
    b1 = ax0.bar(x - w / 2, su_scores, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax0.bar(x + w / 2, base_scores, w, label="PoseLib / OpenCV Baseline", color="#377eb8", edgecolor="black", linewidth=0.6)
    for bar in b1:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f6b4e")
    for bar in b2:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#235784")
    ax0.set_xticks(x)
    ax0.set_xticklabels(tasks, fontsize=9.5, fontweight="bold")
    ax0.set_ylabel("Accuracy (AUC %)", fontsize=10.5, fontweight="bold")
    ax0.set_title("Multi-Dataset Accuracy Across All 5 Geometric Tasks ↑", fontsize=11.5, fontweight="bold")
    ax0.set_ylim(0, 108)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(loc="upper right", frameon=True, fontsize=9.5)

    ax1 = axes[1]
    b1 = ax1.bar(x - w / 2, su_times, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x + w / 2, base_times, w, label="PoseLib / OpenCV Baseline", color="#377eb8", edgecolor="black", linewidth=0.6)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f6b4e")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, color="#235784")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Mean Runtime (ms)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Average Runtime Across All 5 Geometric Tasks (ms) ↓", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(0, max(max(su_times), max(base_times)) * 1.22 + 1.0)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.suptitle(
        "SupeRANSAC Multi-Dataset Benchmark Overview (PhotoTourism, ETH3D, 7Scenes, Kusvod2, AdelaideRMF, Homogr, EVD)",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(assets_dir, "benchmark_overview.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def print_6d_markdown(title, summary, methods, cols):
    print(f"\n### {title}\n")
    print(f"| Dataset | Problems | Method | {cols[0]} ↑ | {cols[1]} ↑ | {cols[2]} ↑ | Med Rot (°) ↓ | Med Trans (cm) ↓ | Mean ms ↓ | Med ms ↓ |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds_name, m_dict in summary.items():
        n_p = m_dict["num_problems"]
        for _, m_label in methods:
            st = m_dict[m_label]
            a0, a1, a2 = st["auc"]
            b = "**" if m_label == "SupeRANSAC" else ""
            print(
                f"| {ds_name} | {n_p} | {b}{m_label}{b} | "
                f"{b}{a0:.2f}%{b} | {b}{a1:.2f}%{b} | {b}{a2:.2f}%{b} | "
                f"{st['med_rot']:.2f} | {st['med_trans']:.2f} | {st['time_ms']:.2f} | {st['med_ms']:.2f} |"
            )


def main():
    parser = argparse.ArgumentParser(description="6D Absolute Pose (PnP) & Rigid Registration (SE(3)) Benchmark")
    parser.add_argument("--problem", choices=["all", "absolute_pose", "rigid_transform"], default="all")
    parser.add_argument("--dataset", choices=["all", "7scenes", "pose6dscene", "rigid_pose_example"], default="all")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    json_path = os.path.join(args.data_dir, "pose_6d_results.json")
    out_json = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                out_json = json.load(f)
        except Exception:
            out_json = {}
    pnp_summary = out_json.get("absolute_pose", {}).get("summary")
    rigid_summary = out_json.get("rigid_transform", {}).get("summary")

    if args.problem in ("all", "absolute_pose"):
        probs = []
        if args.dataset in ("all", "7scenes"):
            probs.extend(load_7scenes_pnp_problems(args.data_dir))
        if args.dataset in ("all", "pose6dscene"):
            probs.extend(load_pose6dscene_pnp_problems(args.data_dir))

        tasks = [(p, 5000, 1000) for p in probs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_eval_pnp_problem, tasks))

        pnp_methods = [("su", "SupeRANSAC"), ("cv", "OpenCV RANSAC")]
        pnp_thresholds = (5.0, 10.0, 20.0)
        pnp_summary = summarize_6d_results(results, pnp_methods, auc_thresholds=pnp_thresholds)
        print_6d_markdown(
            "6D Absolute Pose (PnP) — Multi-Dataset Results",
            pnp_summary,
            pnp_methods,
            ("AUC@(5°,25cm)", "AUC@(10°,50cm)", "AUC@(20°,1m)"),
        )

        pnp_curves = {m_label: [] for _, m_label in pnp_methods}
        for b in BUDGETS:
            min_it = min(b, 1000)
            b_tasks = [(p, b, min_it) for p in probs]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                b_res = list(ex.map(_eval_pnp_problem, b_tasks))
            b_sum = summarize_6d_results(b_res, pnp_methods, auc_thresholds=pnp_thresholds)["Overall"]
            for _, m_label in pnp_methods:
                pnp_curves[m_label].append({
                    "budget": b,
                    "time_ms": b_sum[m_label]["time_ms"],
                    "auc": b_sum[m_label]["auc"],
                })

        if not args.no_plots:
            plot_pnp_figure(pnp_summary, pnp_curves, assets_dir)
        out_json["absolute_pose"] = {"summary": pnp_summary, "curves": pnp_curves}

    if args.problem in ("all", "rigid_transform"):
        probs = []
        if args.dataset in ("all", "7scenes"):
            probs.extend(load_7scenes_rigid_problems(args.data_dir))
        if args.dataset in ("all", "rigid_pose_example"):
            probs.extend(load_rigid_pose_example_problems(args.data_dir))

        tasks = [(p, 5000, 1000) for p in probs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_eval_rigid_problem, tasks))

        rigid_methods = [("su", "SupeRANSAC"), ("cv", "OpenCV RANSAC")]
        rigid_thresholds = (2.5, 5.0, 10.0)
        rigid_summary = summarize_6d_results(results, rigid_methods, auc_thresholds=rigid_thresholds)
        print_6d_markdown(
            "6D Rigid Transformation (3D-3D) — Multi-Dataset Results",
            rigid_summary,
            rigid_methods,
            ("AUC@(2.5°,5cm)", "AUC@(5°,10cm)", "AUC@(10°,20cm)"),
        )

        rigid_curves = {m_label: [] for _, m_label in rigid_methods}
        for b in BUDGETS:
            min_it = min(b, 1000)
            b_tasks = [(p, b, min_it) for p in probs]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                b_res = list(ex.map(_eval_rigid_problem, b_tasks))
            b_sum = summarize_6d_results(b_res, rigid_methods, auc_thresholds=rigid_thresholds)["Overall"]
            for _, m_label in rigid_methods:
                rigid_curves[m_label].append({
                    "budget": b,
                    "time_ms": b_sum[m_label]["time_ms"],
                    "auc": b_sum[m_label]["auc"],
                })

        if not args.no_plots:
            plot_rigid_figure(rigid_summary, results, assets_dir)
        out_json["rigid_transform"] = {"summary": rigid_summary, "curves": rigid_curves}

    if not args.no_plots and pnp_summary is not None and rigid_summary is not None:
        plot_benchmark_overview(args.data_dir, pnp_summary, rigid_summary, assets_dir)

    json_path = os.path.join(args.data_dir, "pose_6d_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
