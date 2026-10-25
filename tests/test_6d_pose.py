"""6D pose benchmark: absolute pose (2D-3D PnP) and rigid registration (3D-3D SE(3)).

Real track (headline): 7Scenes RGB-D, detector matches lifted through the depth maps, GT = dataset poses,
no injected outliers (see datasets/pose6d_problems.py).
Synthetic stress tests (reported separately): pose6dscene outlier sweep, rigid_pose_example tiers.

Baselines
    PnP   : PoseLib estimate_absolute_pose (LO-RANSAC + bundle), OpenCV solvePnPRansac (EPnP, plain RANSAC)
    Rigid : a transparent 3-point Procrustes RANSAC (MSAC scoring, adaptive termination, final LSQ on inliers)
            implemented below in Python/NumPy -- neither OpenCV nor PoseLib ships a rigid-registration RANSAC
            (cv2.estimateAffine3D fits a 12-DoF affine and is not comparable). Being Python, its runtime is
            NOT comparable to the C++ methods and is reported for reference only.

All methods: same threshold (3.5 px / 5 cm), same budget (each method's adaptive termination within 1000-5000
iterations). Where the matches are ordered by matcher confidence (real 7Scenes problems), every method samples
with PROSAC (SupeRANSAC's default; PoseLib progressive_sampling; PROSAC in the Procrustes baseline); the synthetic
problems have no order and use uniform sampling everywhere. PnP speed-accuracy sweep: min_iterations in
SWEEP_MIN_ITERS (max 5000) for SupeRANSAC and PoseLib, the iteration cap in SWEEP_CAPS for OpenCV (no minimum).
Metric: pose error = max(rotation error [deg], 0.5 * translation error [cm]) -> AUC@(2deg,4cm)/(5deg,10cm)/(10deg,20cm).

Usage:
    python tests/test_6d_pose.py --problem all
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

from datasets.benchmark_pairs import DEFAULT_DATA_DIR
from datasets.pose6d_problems import (
    load_7scenes_pnp_problems,
    load_7scenes_rigid_problems,
    load_pose6dscene_pnp_problems,
    load_rigid_pose_example_problems,
)
from evaluation import pose_auc
from benchmark_protocol import (SWEEP_CAPS, SWEEP_MIN_ITERS, TABLE_MAX_ITERS, TABLE_MIN_ITERS, default_workers,
                                per_problem_record, plots_dir)

cv2.setNumThreads(1)

AUC_THRESHOLDS = [2.0, 5.0, 10.0]
TRANS_WEIGHT = 0.5  # deg per cm: 5 deg ~ 10 cm
PNP_METHODS = ["SupeRANSAC", "PoseLib", "OpenCV RANSAC"]
RIGID_METHODS = ["SupeRANSAC", "Procrustes RANSAC"]
COLORS = {"SupeRANSAC": "#1b9e77", "PoseLib": "#377eb8", "OpenCV RANSAC": "#e41a1c", "Procrustes RANSAC": "#e41a1c"}
STYLES = {"SupeRANSAC": "-o", "PoseLib": "--s", "OpenCV RANSAC": ":d", "Procrustes RANSAC": "--s"}


def rotation_error_deg(R_gt, R):
    c = (np.trace(R_gt.T @ R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(R_gt, t_gt, R, t, cm_scale):
    if R is None or t is None or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        return 180.0, 180.0, 1e6
    r = rotation_error_deg(R_gt, R)
    tr = float(np.linalg.norm(np.asarray(t_gt).flatten() - np.asarray(t).flatten()) * cm_scale)
    return float(max(r, TRANS_WEIGHT * tr)), r, tr


# --------------------------------------------------------------------------------------
# Baseline: 3-point Procrustes RANSAC for rigid registration
# --------------------------------------------------------------------------------------
def procrustes(X, Y):
    cx, cy = X.mean(axis=0), Y.mean(axis=0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, cy - R @ cx


class ProsacSampler:
    """PROSAC (Chum & Matas, CVPR 2005) as implemented in PoseLib: samples from a progressively growing
    top-ranked subset of the (confidence-ordered) data, uniform after `max_prosac` samples."""

    def __init__(self, n, k, max_prosac, rng):
        self.N, self.k, self.rng = n, k, rng
        self.T_n = float(max_prosac)
        for i in range(k):
            self.T_n *= (k - i) / (n - i)
        self.T_n_p, self.n, self.t = 1, k, 0

    def sample(self):
        self.t += 1
        if self.t > self.T_n_p and self.n < self.N:
            T_next = self.T_n * (self.n + 1) / (self.n + 1 - self.k)
            self.T_n_p += int(np.ceil(T_next - self.T_n))
            self.T_n = T_next
            self.n += 1
        if self.T_n_p < self.t:
            return self.rng.choice(self.n, self.k, replace=False)
        return np.append(self.rng.choice(self.n - 1, self.k - 1, replace=False), self.n - 1)


def procrustes_ransac(corrs, th, max_iters, min_iters, confidence=0.9999, seed=0, ordered=False):
    """ordered=True: the rows are sorted by matcher confidence and are sampled with PROSAC."""
    rng = np.random.default_rng(seed)
    X, Y = corrs[:, :3], corrs[:, 3:6]
    N = len(X)
    th2 = th * th
    best_score, best_mask = -1.0, None
    it, needed = 0, max_iters
    prosac = ProsacSampler(N, 3, max_iters, rng) if ordered and N > 3 else None
    while it < needed:
        idx = prosac.sample() if prosac is not None else rng.choice(N, 3, replace=False)
        R, t = procrustes(X[idx], Y[idx])
        r2 = np.sum((X @ R.T + t - Y) ** 2, axis=1)
        mask = r2 < th2
        score = float(np.sum(1.0 - r2[mask] / th2))  # MSAC
        if score > best_score:
            best_score, best_mask = score, mask
            w = min(max(mask.mean(), 1e-3), 1.0 - 1e-9)
            needed = int(min(max_iters, max(min_iters, np.ceil(np.log(1 - confidence) / np.log(max(1 - w ** 3, 1e-12))))))
        it += 1
    if best_mask is None or best_mask.sum() < 3:
        return None, None, 0
    R, t = procrustes(X[best_mask], Y[best_mask])
    r2 = np.sum((X @ R.T + t - Y) ** 2, axis=1)
    mask = r2 < th2
    if mask.sum() >= 3:
        R, t = procrustes(X[mask], Y[mask])
    return R, t, int(mask.sum())


# --------------------------------------------------------------------------------------
# Per-problem evaluation
# --------------------------------------------------------------------------------------
def _eval_pnp(args):
    prob, max_iters, min_iters = args[:3]
    methods = args[3] if len(args) > 3 else PNP_METHODS
    ordered = len(prob["probs"]) > 0
    corrs, K, th, cm = prob["corrs"], prob["K"], prob["threshold_px"], prob["cm_scale"]
    R_gt, t_gt = prob["R_gt"], prob["t_gt"]
    pts2d = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
    pts3d = np.ascontiguousarray(corrs[:, 2:5], dtype=np.float64)
    out = {"dataset": prob["dataset"], "detector": prob["detector"], "scene": prob["scene"], "n": len(corrs)}

    # SupeRANSAC (expects a non-negative bounding box: shift the 3D points, un-shift the translation)
    if "SupeRANSAC" not in methods:
        return _eval_pnp_opencv(prob, pts2d, pts3d, K, th, cm, max_iters, out)
    min_c = pts3d.min(axis=0)
    corrs_su = np.ascontiguousarray(np.hstack([pts2d, pts3d - min_c]))
    bbox = list(np.max(corrs_su, axis=0))
    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold = th
    cfg.min_iterations, cfg.max_iterations = min_iters, max_iters
    t0 = time.perf_counter()
    R, t, inl, _, _ = pysuperansac.estimateAbsolutePose(
        corrs_su, pysuperansac.CameraType.SimplePinhole, [float(K[0, 0]), float(K[0, 2]), float(K[1, 2])],
        bbox, prob["probs"] if len(prob["probs"]) else [], config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    if R is None or len(inl) < 4:
        out["SupeRANSAC"] = (180.0, 180.0, 1e6, dt, 0)
    else:
        t = np.asarray(t).flatten() - R @ min_c
        out["SupeRANSAC"] = pose_error(R_gt, t_gt, R, t, cm) + (dt, len(inl))

    # PoseLib
    cam = {"model": "PINHOLE", "width": int(prob["width"]), "height": int(prob["height"]),
           "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}
    ropt = {"max_reproj_error": th, "min_iterations": min_iters, "max_iterations": max_iters}
    if ordered:
        ropt.update({"progressive_sampling": True, "max_prosac_iterations": max_iters})
    t0 = time.perf_counter()
    pose, info = poselib.estimate_absolute_pose(pts2d, pts3d, cam, ropt, {"loss_type": "CAUCHY", "loss_scale": 0.5 * th})
    dt = (time.perf_counter() - t0) * 1000.0
    n_inl = int(np.sum(info.get("inliers", [])))
    if pose is None or n_inl < 4:
        out["PoseLib"] = (180.0, 180.0, 1e6, dt, 0)
    else:
        out["PoseLib"] = pose_error(R_gt, t_gt, pose.R, pose.t, cm) + (dt, n_inl)
    if "OpenCV RANSAC" not in methods:
        return out
    return _eval_pnp_opencv(prob, pts2d, pts3d, K, th, cm, max_iters, out)


def _eval_pnp_opencv(prob, pts2d, pts3d, K, th, cm, max_iters, out):
    """OpenCV solvePnPRansac (EPnP, uniform sampling, no minimum iteration count): max_iters is its cap."""
    R_gt, t_gt = prob["R_gt"], prob["t_gt"]
    t0 = time.perf_counter()
    ok, rvec, tvec, inl_cv = cv2.solvePnPRansac(pts3d.astype(np.float32), pts2d.astype(np.float32), K.astype(np.float64), None,
                                                iterationsCount=max_iters, reprojectionError=float(th), confidence=0.9999,
                                                flags=cv2.SOLVEPNP_EPNP)
    dt = (time.perf_counter() - t0) * 1000.0
    n_cv = 0 if (not ok or inl_cv is None) else len(inl_cv)
    if not ok or n_cv < 4:
        out["OpenCV RANSAC"] = (180.0, 180.0, 1e6, dt, 0)
    else:
        out["OpenCV RANSAC"] = pose_error(R_gt, t_gt, cv2.Rodrigues(rvec)[0], tvec.flatten(), cm) + (dt, n_cv)
    return out


def _eval_rigid(args):
    prob, max_iters, min_iters = args
    corrs, th, cm = prob["corrs"], prob["threshold_3d"], prob["cm_scale"]
    R_gt, t_gt = prob["R_gt"], prob["t_gt"]
    out = {"dataset": prob["dataset"], "detector": prob["detector"], "scene": prob["scene"], "n": len(corrs)}

    # SupeRANSAC (non-negative bounding box; undo the shift afterwards)
    mn = corrs.min(axis=0)
    shifted = np.ascontiguousarray(corrs - mn)
    bbox = np.ascontiguousarray(shifted.max(axis=0) + 1e-3)
    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold = th
    cfg.min_iterations, cfg.max_iterations = min_iters, max_iters
    t0 = time.perf_counter()
    T, inl, _, _ = pysuperansac.estimateRigidTransform(shifted, bbox, prob["probs"] if len(prob["probs"]) else [], config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    if T is None or len(inl) < 3 or not np.all(np.isfinite(T)):
        out["SupeRANSAC"] = (180.0, 180.0, 1e6, dt, 0)
    else:
        # SupeRANSAC returns T such that [x 1] @ T maps shifted source to shifted destination.
        T1 = np.eye(4); T1[3, :3] = -mn[:3]
        T2 = np.eye(4); T2[3, :3] = mn[3:]
        M = (T1 @ T @ T2).T
        out["SupeRANSAC"] = pose_error(R_gt, t_gt, M[:3, :3], M[:3, 3], cm) + (dt, len(inl))

    t0 = time.perf_counter()
    R, t, n_inl = procrustes_ransac(corrs, th, max_iters, min_iters, ordered=len(prob["probs"]) > 0)
    dt = (time.perf_counter() - t0) * 1000.0
    out["Procrustes RANSAC"] = (pose_error(R_gt, t_gt, R, t, cm) if R is not None else (180.0, 180.0, 1e6)) + (dt, n_inl)
    return out


# --------------------------------------------------------------------------------------
# Aggregation / output
# --------------------------------------------------------------------------------------
def summarize(results, methods, group_key="dataset"):
    groups = {}
    for r in results:
        groups.setdefault(r[group_key], []).append(r)
    groups["Overall"] = results
    summary = {}
    for g, rs in groups.items():
        summary[g] = {"num_problems": len(rs)}
        for m in methods:
            summary[g][m] = {
                "auc": [100.0 * a for a in pose_auc([r[m][0] for r in rs], AUC_THRESHOLDS)],
                "med_rot": float(np.median([r[m][1] for r in rs])),
                "med_trans": float(np.median([r[m][2] for r in rs])),
                "time_ms": float(np.mean([r[m][3] for r in rs])),
                "inliers": float(np.mean([r[m][4] for r in rs])),
            }
        summary[g]["per_pair"] = per_problem_record(rs, methods)
    return summary


def print_markdown(title, summary, methods, group_label="Dataset"):
    print(f"\n### {title}\n")
    print(f"| {group_label} | Problems | Method | AUC@(2°,4cm) | AUC@(5°,10cm) | AUC@(10°,20cm) | Med rot (°) | Med trans (cm) | Mean ms |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for g, md in summary.items():
        best = [max(md[m]["auc"][i] for m in methods) for i in range(3)]
        for m in methods:
            st = md[m]
            cells = [f"**{v:.2f}%**" if abs(v - b) < 1e-9 else f"{v:.2f}%" for v, b in zip(st["auc"], best)]
            print(f"| {g} | {md['num_problems']} | {m} | {cells[0]} | {cells[1]} | {cells[2]} | "
                  f"{st['med_rot']:.2f} | {st['med_trans']:.2f} | {st['time_ms']:.1f} |")


def plot_task(title, by_dataset, by_detector, curves, methods, out_path):
    fig, axes = plt.subplots(1, 3 if curves else 2, figsize=(16 if curves else 11, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    w = 0.8 / len(methods)
    for ax, summary, sub in ((axes[0], by_dataset, "by dataset"), (axes[1], by_detector, "7Scenes RGB-D by detector")):
        names = [k for k in summary if k != "Overall"] + (["Overall"] if sub == "by dataset" else [])
        x = np.arange(len(names))
        for i, m in enumerate(methods):
            vals = [summary[n][m]["auc"][1] for n in names]
            bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=m, color=COLORS[m], edgecolor="black", linewidth=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.0f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (synthetic outliers)", "\n(synthetic)") for n in names], fontsize=8)
        ax.set_ylabel("AUC@(5°,10cm) (%)")
        ax.set_title(f"{title} -- {sub}", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="upper left")
    if not curves:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out_path}")
        return
    ax = axes[2]
    for m in methods:
        pts = curves[m]
        ax.plot([p["time_ms"] for p in pts], [p["auc"][1] for p in pts], STYLES[m], color=COLORS[m], linewidth=2.2, markersize=6, label=m)
        star = [p for p in pts if p.get("min_iterations", TABLE_MIN_ITERS) == TABLE_MIN_ITERS and p.get("max_iterations", TABLE_MAX_ITERS) == TABLE_MAX_ITERS]
        ax.plot([p["time_ms"] for p in star], [p["auc"][1] for p in star], "*", color=COLORS[m], markersize=16, markeredgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("Average runtime (ms, log)")
    ax.set_ylabel("AUC@(5°,10cm) (%)")
    ax.set_title(f"{title} -- speed/accuracy, 7Scenes RGB-D", fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def run_task(name, real_probs, synth_probs, eval_fn, methods, workers, assets_dir, no_plots, sweep=False):
    probs = real_probs + synth_probs
    print(f"[{name}] real 7Scenes problems: {len(real_probs)}, synthetic stress problems: {len(synth_probs)}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(eval_fn, [(p, TABLE_MAX_ITERS, TABLE_MIN_ITERS) for p in probs]))
    real_res = [r for r in results if r["dataset"] == "7Scenes RGB-D"]
    by_dataset = summarize(results, methods)
    by_detector = summarize(real_res, methods, group_key="detector")
    by_detector.pop("Overall", None)
    synth_by_level = summarize([r for r in results if r["dataset"] != "7Scenes RGB-D"], methods, group_key="detector") if synth_probs else {}
    print_markdown(f"{name} -- by dataset", by_dataset, methods)
    print_markdown(f"{name} -- 7Scenes RGB-D (real) by detector", by_detector, methods, "Detector")
    if synth_probs:
        print_markdown(f"{name} -- synthetic stress tests by level", synth_by_level, methods, "Level")

    curves = {}
    if sweep:  # PnP only: min_iterations sweep for SupeRANSAC / PoseLib, iteration-cap sweep for OpenCV
        curves = {m: [] for m in methods}
        lib = [m for m in methods if m != "OpenCV RANSAC"]
        for mi in SWEEP_MIN_ITERS:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                b_res = list(ex.map(eval_fn, [(p, TABLE_MAX_ITERS, mi, lib) for p in real_probs]))
            s = summarize(b_res, lib)["Overall"]
            for m in lib:
                curves[m].append({"min_iterations": mi, "time_ms": s[m]["time_ms"], "auc": s[m]["auc"]})
        if "OpenCV RANSAC" in methods:
            for cap in SWEEP_CAPS:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    b_res = list(ex.map(eval_fn, [(p, cap, 0, ["OpenCV RANSAC"]) for p in real_probs]))
                s = summarize(b_res, ["OpenCV RANSAC"])["Overall"]
                curves["OpenCV RANSAC"].append({"max_iterations": cap, "time_ms": s["OpenCV RANSAC"]["time_ms"],
                                                "auc": s["OpenCV RANSAC"]["auc"]})
    if not no_plots:
        fname = "absolute_pose_experiments.png" if "Absolute" in name else "rigid_transform_experiments.png"
        plot_task(name, by_dataset, by_detector, curves, methods, os.path.join(assets_dir, fname))
    return {"by_dataset": by_dataset, "by_detector": by_detector, "synthetic_by_level": synth_by_level, "curves": curves}


def main():
    parser = argparse.ArgumentParser(description="6D pose benchmark (PnP & rigid registration)")
    parser.add_argument("--problem", choices=["all", "absolute_pose", "rigid_transform"], default="all")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    assets_dir = plots_dir(args.data_dir)
    json_path = os.path.join(args.data_dir, "pose_6d_results.json")
    out_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            out_json = json.load(f)

    if args.problem in ("all", "absolute_pose"):
        out_json["absolute_pose"] = run_task("6D Absolute Pose (PnP)", load_7scenes_pnp_problems(args.data_dir),
                                             load_pose6dscene_pnp_problems(), _eval_pnp, PNP_METHODS, args.workers, assets_dir, args.no_plots,
                                             sweep=True)
    if args.problem in ("all", "rigid_transform"):
        out_json["rigid_transform"] = run_task("6D Rigid Registration (SE(3))", load_7scenes_rigid_problems(args.data_dir),
                                               load_rigid_pose_example_problems(), _eval_rigid, RIGID_METHODS, args.workers, assets_dir, args.no_plots)

    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=1)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
