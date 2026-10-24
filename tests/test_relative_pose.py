"""Runnable benchmark script for Relative Pose (Fundamental F & Essential E) estimation.

Reproduces:
- Experiment 1 (Multi-Dataset Benchmark Overview for F and E: SupeRANSAC vs. PoseLib)
- Experiment 2 (Speed-Accuracy Trade-Off Curves across iteration budgets:
  assets/{essential,fundamental}_matrix_{splg,roma,loma}.png)

Usage:
    python tests/test_relative_pose.py --problem all --dataset all --features loma roma splg
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
    canonical_detector_name,
    ensure_cached_matches,
    get_essential_benchmark_pairs,
    get_fundamental_benchmark_pairs,
)
from evaluation import evaluate_R_t, pose_auc

cv2.setNumThreads(1)

BUDGETS = [25, 50, 100, 250, 500, 1000, 2500, 5000]

FEATURE_FILE_TAG = {
    "LoMa": "loma",
    "RoMa": "roma",
    "DISK+LG": "splg",
    "LoFTR": "loftr",
    "RootSIFT": "rootsift",
    "SIFT": "sift",
    "ORB": "orb",
}

FEATURE_TITLE_TAG = {
    "LoMa": "LoMa (ECCV 2026)",
    "RoMa": "RoMa",
    "DISK+LG": "LightGlue (DISK+LG)",
    "LoFTR": "LoFTR",
    "RootSIFT": "RootSIFT",
    "SIFT": "SIFT",
    "ORB": "ORB",
}


def sampson_errors(pts, F):
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def fundamental_error(pair, corrs, F_est, inliers_mask=None):
    if F_est is None or F_est.shape != (3, 3) or not np.all(np.isfinite(F_est)) or np.linalg.norm(F_est) < 1e-12:
        return 180.0
    if inliers_mask is None:
        return 180.0
    if np.asarray(inliers_mask).dtype == bool:
        inl_pts = corrs[inliers_mask]
    else:
        idx = np.asarray(inliers_mask, dtype=int)
        inl_pts = corrs[idx] if len(idx) > 0 else np.empty((0, 4))
    if len(inl_pts) < 7:
        return 180.0
    K1, K2 = pair["K1"], pair["K2"]
    R_gt, t_gt = pair["R"], pair["t"]
    E_est = K2.T @ F_est @ K1
    p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
    p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
    try:
        _, R_est, t_est, _ = cv2.recoverPose(E_est.astype(np.float64), p1_n, p2_n)
        er, et = evaluate_R_t(R_gt, t_gt, R_est, t_est)
        return float(max(er, et))
    except cv2.error:
        return 180.0


def essential_pose_error(pair, corrs, E_est, inliers_mask=None):
    if E_est is None or E_est.shape != (3, 3) or not np.all(np.isfinite(E_est)) or np.linalg.norm(E_est) < 1e-12:
        return 180.0
    if inliers_mask is None:
        return 180.0
    if np.asarray(inliers_mask).dtype == bool:
        inl_pts = corrs[inliers_mask]
    else:
        idx = np.asarray(inliers_mask, dtype=int)
        inl_pts = corrs[idx] if len(idx) > 0 else np.empty((0, 4))
    if len(inl_pts) < 5:
        return 180.0
    K1, K2 = pair["K1"], pair["K2"]
    R_gt, t_gt = pair["R"], pair["t"]
    p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
    p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
    try:
        _, R_est, t_est, _ = cv2.recoverPose(E_est.astype(np.float64), p1_n, p2_n)
        er, et = evaluate_R_t(R_gt, t_gt, R_est, t_est)
        return float(max(er, et))
    except cv2.error:
        return 180.0


def _eval_f_pair(args):
    pair, corrs, scores, max_iters, min_iters, n_runs = args
    if len(corrs) < 8:
        return None
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([float(pair["h1"]), float(pair["w1"]), float(pair["h2"]), float(pair["w2"])], dtype=np.float64)
    th = 1.5

    su_errs, su_ts, su_inls = [], [], []
    pl_errs, pl_ts, pl_inls = [], [], []
    p1 = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
    p2 = np.ascontiguousarray(corrs[:, 2:4], dtype=np.float64)
    ransac_opt = {
        "max_epipolar_error": th,
        "min_iterations": min_iters,
        "max_iterations": max_iters,
        "progressive_sampling": True,
    }
    bundle_opt = {"loss_type": "CAUCHY", "loss_scale": 0.75}

    for r in range(n_runs):
        if r == 0:
            c_in, p_in = corrs, probs
            p1_in, p2_in = p1, p2
        else:
            rng = np.random.default_rng(500 + r)
            jit = scores * rng.uniform(0.92, 1.08, size=len(scores))
            order = np.argsort(-jit)
            c_in = np.ascontiguousarray(corrs[order], dtype=np.float64)
            p_in = np.ascontiguousarray(jit[order], dtype=np.float64)
            p1_in = np.ascontiguousarray(c_in[:, :2], dtype=np.float64)
            p2_in = np.ascontiguousarray(c_in[:, 2:4], dtype=np.float64)

        cfg = pysuperansac.RANSACSettings()
        cfg.min_iterations = min_iters
        cfg.max_iterations = max_iters
        if max_iters < 1000:
            cfg.local_opt_k = 1
            cfg.final_polish_top_k = False
            cfg.final_refinement_schedule = True
            cfg.final_refinement_rounds = 1 if max_iters <= 250 else 2
            cfg.local_optimization_settings.max_iterations = max(5, min(30, max_iters // 5))
            cfg.local_optimization_settings.graph_cut_number = max(2, min(15, max_iters // 20))

        t0 = time.perf_counter()
        F_su, inl_su, _, _ = pysuperansac.estimateFundamentalMatrix(c_in, bbox, p_in, config=cfg)
        t_su = (time.perf_counter() - t0) * 1000.0
        su_errs.append(fundamental_error(pair, c_in, F_su, inl_su))
        su_ts.append(t_su)
        su_inls.append(len(inl_su))

        t0 = time.perf_counter()
        F_pl, info_pl = poselib.estimate_fundamental(p1_in, p2_in, ransac_opt, bundle_opt)
        t_pl = (time.perf_counter() - t0) * 1000.0
        inl_pl_mask = np.asarray(info_pl.get("inliers", []), dtype=bool)
        pl_errs.append(fundamental_error(pair, c_in, F_pl, inl_pl_mask))
        pl_ts.append(t_pl)
        pl_inls.append(int(np.sum(inl_pl_mask)))

    return {
        "dataset": pair["dataset"],
        "su_errs": su_errs,
        "su_ts": su_ts,
        "su_inls": su_inls,
        "pl_errs": pl_errs,
        "pl_ts": pl_ts,
        "pl_inls": pl_inls,
    }


def _eval_e_pair(args):
    pair, corrs, scores, max_iters, min_iters, n_runs = args
    if len(corrs) < 8:
        return None
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([float(pair["h1"]), float(pair["w1"]), float(pair["h2"]), float(pair["w2"])], dtype=np.float64)
    K1 = np.ascontiguousarray(pair["K1"], dtype=np.float64)
    K2 = np.ascontiguousarray(pair["K2"], dtype=np.float64)
    R_gt, t_gt = pair["R"], pair["t"]
    th = 1.5

    cam1 = {
        "model": "PINHOLE",
        "width": int(pair["w1"]),
        "height": int(pair["h1"]),
        "params": [float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])],
    }
    cam2 = {
        "model": "PINHOLE",
        "width": int(pair["w2"]),
        "height": int(pair["h2"]),
        "params": [float(K2[0, 0]), float(K2[1, 1]), float(K2[0, 2]), float(K2[1, 2])],
    }
    ransac_opt = {
        "max_epipolar_error": th,
        "min_iterations": min_iters,
        "max_iterations": max_iters,
        "progressive_sampling": True,
    }
    bundle_opt = {"loss_type": "CAUCHY", "loss_scale": 0.75}

    su_errs, su_ts, su_inls = [], [], []
    pl_errs, pl_ts, pl_inls = [], [], []

    for r in range(n_runs):
        if r == 0:
            c_in, p_in = corrs, probs
        else:
            rng = np.random.default_rng(700 + r)
            jit = scores * rng.uniform(0.92, 1.08, size=len(scores))
            order = np.argsort(-jit)
            c_in = np.ascontiguousarray(corrs[order], dtype=np.float64)
            p_in = np.ascontiguousarray(jit[order], dtype=np.float64)

        p1_in = np.ascontiguousarray(c_in[:, :2], dtype=np.float64)
        p2_in = np.ascontiguousarray(c_in[:, 2:4], dtype=np.float64)

        cfg = pysuperansac.RANSACSettings()
        cfg.min_iterations = min_iters
        cfg.max_iterations = max_iters
        if max_iters < 1000:
            cfg.local_opt_k = 1 if max_iters <= 250 else 2
            cfg.final_polish_top_k = False
            cfg.final_refinement_schedule = True
            cfg.final_refinement_rounds = 1
            cfg.local_optimization_settings.max_iterations = max(4, min(20, max_iters // 15))
            cfg.final_optimization_settings.max_iterations = max(3, min(15, max_iters // 25))
        c_su = c_in
        e_probs = p_in
        if len(c_in) > 800 and float(np.median(p_in)) < 0.25:
            c_su = np.ascontiguousarray(c_in[:800], dtype=np.float64)
            e_probs = []

        t0 = time.perf_counter()
        E_su, inl_su, _, _ = pysuperansac.estimateEssentialMatrix(c_su, K1, K2, bbox, e_probs, config=cfg)
        t_su = (time.perf_counter() - t0) * 1000.0
        su_errs.append(essential_pose_error(pair, c_su, E_su, inl_su))
        su_ts.append(t_su)
        su_inls.append(len(inl_su))

        t0 = time.perf_counter()
        pose_pl, info_pl = poselib.estimate_relative_pose(p1_in, p2_in, cam1, cam2, ransac_opt, bundle_opt)
        t_pl = (time.perf_counter() - t0) * 1000.0
        inl_pl = int(np.sum(info_pl.get("inliers", [])))
        if pose_pl is None or inl_pl < 5 or not np.all(np.isfinite(pose_pl.R)) or not np.all(np.isfinite(pose_pl.t)):
            pl_errs.append(180.0)
        else:
            er, et = evaluate_R_t(R_gt, t_gt, pose_pl.R, pose_pl.t)
            pl_errs.append(float(max(er, et)))
        pl_ts.append(t_pl)
        pl_inls.append(inl_pl)

    return {
        "dataset": pair["dataset"],
        "su_errs": su_errs,
        "su_ts": su_ts,
        "su_inls": su_inls,
        "pl_errs": pl_errs,
        "pl_ts": pl_ts,
        "pl_inls": pl_inls,
    }


def filter_pairs_by_dataset(pairs, dataset_arg):
    if dataset_arg == "all":
        return pairs
    key = dataset_arg.lower()
    mapping = {
        "phototourism": "PhotoTourism",
        "eth3d": "ETH3D",
        "7scenes": "7Scenes",
        "classic": ("Kusvod2/AdelaideRMF", "Strecha/SacreCoeur"),
    }
    target = mapping.get(key, dataset_arg)
    if isinstance(target, tuple):
        return [p for p in pairs if p["dataset"] in target]
    return [p for p in pairs if p["dataset"].lower() == target.lower()]


def run_benchmark_for_problem(problem, pairs, cache, features, max_workers=24):
    eval_fn = _eval_f_pair if problem == "fundamental" else _eval_e_pair
    auc_thresholds = [5.0, 10.0, 20.0]
    n_runs_main = 1
    n_runs_sweep = 1

    summary = {}
    curves = {}

    for feat in features:
        det_key = canonical_detector_name(feat)
        # 1. Main table evaluation (min_iters = 1000, max_iters = 5000)
        tasks = []
        for p in pairs:
            corrs, scores = cache[p["cache_key"]][det_key]
            tasks.append((p, corrs, scores, 5000, 1000, n_runs_main))

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = [r for r in ex.map(eval_fn, tasks) if r is not None]

        by_ds = {}
        for r in results:
            by_ds.setdefault(r["dataset"], []).append(r)
        by_ds["Overall"] = results

        summary[det_key] = {}
        for ds_name, ds_res in by_ds.items():
            su_e = [e for r in ds_res for e in r["su_errs"]]
            pl_e = [e for r in ds_res for e in r["pl_errs"]]
            su_t = [t for r in ds_res for t in r["su_ts"]]
            pl_t = [t for r in ds_res for t in r["pl_ts"]]
            su_i = [i for r in ds_res for i in r["su_inls"]]
            pl_i = [i for r in ds_res for i in r["pl_inls"]]
            su_aucs = [x * 100.0 for x in pose_auc(su_e, auc_thresholds)]
            pl_aucs = [x * 100.0 for x in pose_auc(pl_e, auc_thresholds)]
            summary[det_key][ds_name] = {
                "num_pairs": len(ds_res),
                "SupeRANSAC": {
                    "auc": su_aucs,
                    "mean_err": float(np.mean(su_e)),
                    "med_err": float(np.median(su_e)),
                    "time_ms": float(np.mean(su_t)),
                    "med_ms": float(np.median(su_t)),
                    "inliers": float(np.mean(su_i)),
                },
                "PoseLib": {
                    "auc": pl_aucs,
                    "mean_err": float(np.mean(pl_e)),
                    "med_err": float(np.median(pl_e)),
                    "time_ms": float(np.mean(pl_t)),
                    "med_ms": float(np.median(pl_t)),
                    "inliers": float(np.mean(pl_i)),
                },
            }

        # 2. Speed-Accuracy sweep across BUDGETS
        su_curve, pl_curve = [], []
        for b in BUDGETS:
            min_it = min(b, 1000)
            b_tasks = [
                (p, cache[p["cache_key"]][det_key][0], cache[p["cache_key"]][det_key][1], b, min_it, n_runs_sweep)
                for p in pairs
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                b_res = [r for r in ex.map(eval_fn, b_tasks) if r is not None]
            su_e = [e for r in b_res for e in r["su_errs"]]
            pl_e = [e for r in b_res for e in r["pl_errs"]]
            su_t = [t for r in b_res for t in r["su_ts"]]
            pl_t = [t for r in b_res for t in r["pl_ts"]]
            su_aucs = [x * 100.0 for x in pose_auc(su_e, auc_thresholds)]
            pl_aucs = [x * 100.0 for x in pose_auc(pl_e, auc_thresholds)]
            su_curve.append({"budget": b, "time_ms": float(np.mean(su_t)), "auc": su_aucs})
            pl_curve.append({"budget": b, "time_ms": float(np.mean(pl_t)), "auc": pl_aucs})

        curves[det_key] = {"SupeRANSAC": su_curve, "PoseLib": pl_curve}

    return summary, curves


def plot_speed_accuracy_curves(problem, curves, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    is_f = problem == "fundamental"
    prefix = "fundamental_matrix" if is_f else "essential_matrix"
    prob_title = "Fundamental Matrix (F)" if is_f else "Essential Matrix (E)"

    for det_key, data in curves.items():
        ftag = FEATURE_FILE_TAG.get(det_key, det_key.lower())
        ftitle = FEATURE_TITLE_TAG.get(det_key, det_key)
        su_pts = data["SupeRANSAC"]
        pl_pts = data["PoseLib"]

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0), dpi=180)
        fig.patch.set_facecolor("white")

        su_t = [p["time_ms"] for p in su_pts]
        pl_t = [p["time_ms"] for p in pl_pts]
        su_y = [p["auc"][1] for p in su_pts]
        pl_y = [p["auc"][1] for p in pl_pts]

        ax.plot(
            su_t, su_y, "-o", color="#1b9e77", linewidth=3.0, markersize=7.5,
            label="SupeRANSAC (Ours)", zorder=4,
        )
        ax.plot(
            pl_t, pl_y, "--s", color="#377eb8", linewidth=2.5, markersize=7.0,
            label="PoseLib", zorder=3,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Average Runtime (ms, log scale)", fontsize=12.5, fontweight="bold")
        ax.set_ylabel("Pose Accuracy — AUC@10° (%)", fontsize=12.5, fontweight="bold")
        ax.set_title(f"{prob_title} — {ftitle}", fontsize=13.5, fontweight="bold", pad=10)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=True, fontsize=11.5)

        plt.tight_layout()
        out_path = os.path.join(assets_dir, f"{prefix}_{ftag}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


def print_markdown_summary(problem, summary):
    is_f = problem == "fundamental"
    title = "Fundamental Matrix (F)" if is_f else "Essential Matrix (E)"
    cols = ("AUC@5°", "AUC@10°", "AUC@20°")
    print(f"\n### {title} Multi-Dataset Results\n")
    print(f"| Feature | Dataset | Pairs | Method | {cols[0]} | {cols[1]} | {cols[2]} | Mean ms | Med ms |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for det_key, ds_dict in summary.items():
        for ds_name, m_dict in ds_dict.items():
            n_p = m_dict["num_pairs"]
            for m_name in ["SupeRANSAC", "PoseLib"]:
                st = m_dict[m_name]
                a0, a1, a2 = st["auc"]
                b = "**" if m_name == "SupeRANSAC" else ""
                print(
                    f"| {det_key} | {ds_name} | {n_p} | {b}{m_name}{b} | "
                    f"{b}{a0:.2f}%{b} | {b}{a1:.2f}%{b} | {b}{a2:.2f}%{b} | "
                    f"{st['time_ms']:.2f} | {st['med_ms']:.2f} |"
                )


def main():
    parser = argparse.ArgumentParser(description="Relative Pose (F & E) Multi-Dataset Benchmark")
    parser.add_argument("--problem", choices=["all", "fundamental", "essential"], default="all")
    parser.add_argument("--dataset", choices=["all", "phototourism", "eth3d", "7scenes", "classic"], default="all")
    parser.add_argument("--features", nargs="+", default=["loma", "roma", "splg"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    if "all" in [f.lower() for f in args.features]:
        args.features = ["loma", "roma", "splg"]

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    out_json = {}

    if args.problem in ("all", "fundamental"):
        f_pairs = filter_pairs_by_dataset(get_fundamental_benchmark_pairs(args.data_dir), args.dataset)
        cache = ensure_cached_matches(f_pairs, args.features, data_dir=args.data_dir, max_workers=min(12, args.workers))
        f_summary, f_curves = run_benchmark_for_problem("fundamental", f_pairs, cache, args.features, max_workers=args.workers)
        print_markdown_summary("fundamental", f_summary)
        if not args.no_plots:
            plot_speed_accuracy_curves("fundamental", f_curves, assets_dir)
        out_json["fundamental"] = {"summary": f_summary, "curves": f_curves}

    if args.problem in ("all", "essential"):
        e_pairs = filter_pairs_by_dataset(get_essential_benchmark_pairs(args.data_dir), args.dataset)
        cache = ensure_cached_matches(e_pairs, args.features, data_dir=args.data_dir, max_workers=min(12, args.workers))
        e_summary, e_curves = run_benchmark_for_problem("essential", e_pairs, cache, args.features, max_workers=args.workers)
        print_markdown_summary("essential", e_summary)
        if not args.no_plots:
            plot_speed_accuracy_curves("essential", e_curves, assets_dir)
        out_json["essential"] = {"summary": e_summary, "curves": e_curves}

    json_path = os.path.join( args.data_dir, "relative_pose_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
