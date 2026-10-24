"""Runnable benchmark script for Homography (H) estimation across multiple datasets.

Reproduces:
- Experiment 1 (Multi-Dataset Benchmark Overview for Homography H: SupeRANSAC vs. PoseLib)
- Experiment 2 (Speed-Accuracy Trade-Off Curves across iteration budgets:
  assets/homography_matrix_{splg,roma,loma}.png)

Usage:
    python tests/test_homography.py --dataset all --features loma roma splg
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
    get_homography_benchmark_pairs,
)
from evaluation import pose_auc

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


def prepare_homography_gt(pair, cache):
    """Ensure pair has H_gt and gt_inliers (for 3D datasets, extract dominant ground-truth plane)."""
    if pair.get("H_gt") is not None and pair.get("gt_inliers") is not None and len(pair["gt_inliers"]) >= 8:
        return pair

    # Pool high-confidence epipolar-consistent correspondences across cached detectors to find dominant scene plane
    k = pair["cache_key"]
    pooled = []
    for det_data in cache.get(k, {}).values():
        corrs, _ = det_data
        if len(corrs) >= 8:
            if pair.get("F_gt") is not None:
                res = _sampson_errors(corrs, pair["F_gt"])
                inl = corrs[res < 2.0]
                if len(inl) >= 8:
                    pooled.append(inl)
            else:
                pooled.append(corrs)
    if not pooled:
        return None
    all_pts = np.vstack(pooled)
    if len(all_pts) < 16:
        return None
    H_init, inl_mask = cv2.findHomography(all_pts[:, :2], all_pts[:, 2:4], cv2.USAC_MAGSAC, 2.0, maxIters=10000)
    if H_init is None or inl_mask is None or np.sum(inl_mask) < 16:
        return None
    plane_pts = all_pts[inl_mask.ravel() == 1]
    H_gt, _ = cv2.findHomography(plane_pts[:, :2], plane_pts[:, 2:4], 0)
    if H_gt is None or not np.all(np.isfinite(H_gt)):
        return None
    out = dict(pair)
    out["H_gt"] = H_gt
    out["gt_inliers"] = plane_pts
    return out


def homography_transfer_error(pair, H_est, num_inliers=4):
    if num_inliers < 4 or H_est is None or H_est.shape != (3, 3) or not np.all(np.isfinite(H_est)) or np.linalg.norm(H_est) < 1e-12:
        return 1e6
    gt_inl = pair["gt_inliers"]
    mn = np.min(gt_inl[:, :2], axis=0)
    mx = np.max(gt_inl[:, :2], axis=0)
    if np.any(mx - mn < 20.0):
        mn = np.array([0.2 * float(pair["w1"]), 0.2 * float(pair["h1"])], dtype=np.float64)
        mx = np.array([0.8 * float(pair["w1"]), 0.8 * float(pair["h1"])], dtype=np.float64)
    corners = np.array([
        [mn[0], mn[1], 1.0],
        [mx[0], mn[1], 1.0],
        [mx[0], mx[1], 1.0],
        [mn[0], mx[1], 1.0],
    ], dtype=np.float64)
    c_gt = corners @ pair["H_gt"].T
    if np.any(np.abs(c_gt[:, 2]) < 1e-8):
        return 1e6
    c_gt = c_gt[:, :2] / c_gt[:, 2:3]
    c_est = corners @ H_est.T
    if np.any(np.abs(c_est[:, 2]) < 1e-8):
        return 1e6
    c_est = c_est[:, :2] / c_est[:, 2:3]
    return float(np.mean(np.linalg.norm(c_est - c_gt, axis=1)))


def _eval_h_pair(args):
    pair, corrs, scores, max_iters, min_iters, n_runs = args
    if len(corrs) < 4:
        return None
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([float(pair["h1"]), float(pair["w1"]), float(pair["h2"]), float(pair["w2"])], dtype=np.float64)

    ransac_opt = {
        "max_reproj_error": 4.0,
        "min_iterations": min_iters,
        "max_iterations": max_iters,
        "progressive_sampling": True,
    }
    bundle_opt = {"loss_type": "CAUCHY", "loss_scale": 2.0}

    su_errs, su_ts, su_inls = [], [], []
    pl_errs, pl_ts, pl_inls = [], [], []

    for r in range(n_runs):
        if r == 0:
            c_in, p_in = corrs, probs
        else:
            rng = np.random.default_rng(900 + r)
            jit = scores * rng.uniform(0.90, 1.10, size=len(scores))
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
            cfg.local_optimization_settings.max_iterations = max(5, min(30, max_iters // 5))
            cfg.local_optimization_settings.graph_cut_number = max(2, min(15, max_iters // 20))

        t0 = time.perf_counter()
        H_su, inl_su, _, _ = pysuperansac.estimateHomography(c_in, bbox, p_in, config=cfg)
        t_su = (time.perf_counter() - t0) * 1000.0
        su_errs.append(homography_transfer_error(pair, H_su, len(inl_su)))
        su_ts.append(t_su)
        su_inls.append(len(inl_su))

        t0 = time.perf_counter()
        H_pl, info_pl = poselib.estimate_homography(p1_in, p2_in, ransac_opt, bundle_opt)
        t_pl = (time.perf_counter() - t0) * 1000.0
        n_inl_pl = int(np.sum(info_pl.get("inliers", [])))
        pl_errs.append(homography_transfer_error(pair, H_pl, n_inl_pl))
        pl_ts.append(t_pl)
        pl_inls.append(n_inl_pl)

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
        "homogr": "Homogr",
        "evd": "EVD",
        "adelaidermf": "AdelaideRMF/Multi-H",
        "phototourism": "PhotoTourism",
        "eth3d": "ETH3D",
        "7scenes": "7Scenes",
    }
    target = mapping.get(key, dataset_arg)
    return [p for p in pairs if p["dataset"].lower() == target.lower()]


def run_homography_benchmark(pairs, cache, features, max_workers=24):
    auc_thresholds = [2.5, 5.0, 10.0]
    n_runs_main = 1
    n_runs_sweep = 1

    valid_pairs = []
    for p in pairs:
        hp = prepare_homography_gt(p, cache)
        if hp is not None:
            valid_pairs.append(hp)

    summary = {}
    curves = {}

    for feat in features:
        det_key = canonical_detector_name(feat)
        tasks = [
            (p, cache[p["cache_key"]][det_key][0], cache[p["cache_key"]][det_key][1], 5000, 1000, n_runs_main)
            for p in valid_pairs
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = [r for r in ex.map(_eval_h_pair, tasks) if r is not None]

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

        su_curve, pl_curve = [], []
        for b in BUDGETS:
            min_it = min(b, 1000)
            b_tasks = [
                (p, cache[p["cache_key"]][det_key][0], cache[p["cache_key"]][det_key][1], b, min_it, n_runs_sweep)
                for p in valid_pairs
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                b_res = [r for r in ex.map(_eval_h_pair, b_tasks) if r is not None]
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


def plot_homography_curves(curves, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)

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
        ax.set_ylabel("Corner Accuracy — AUC@5.0 px (%)", fontsize=12.5, fontweight="bold")
        ax.set_title(f"Homography (H) — {ftitle}", fontsize=13.5, fontweight="bold", pad=10)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=True, fontsize=11.5)

        plt.tight_layout()
        out_path = os.path.join(assets_dir, f"homography_matrix_{ftag}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


def print_markdown_summary(summary):
    print("\n### Homography (H) Multi-Dataset Results\n")
    print("| Feature | Dataset | Pairs | Method | AUC@2.5px | AUC@5px | AUC@10px | Mean ms | Med ms |")
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
    parser = argparse.ArgumentParser(description="Homography (H) Multi-Dataset Benchmark")
    parser.add_argument(
        "--dataset",
        choices=["all", "homogr", "evd", "adelaidermf", "phototourism", "eth3d", "7scenes"],
        default="all",
    )
    parser.add_argument("--features", nargs="+", default=["loma", "roma", "splg"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    if "all" in [f.lower() for f in args.features]:
        args.features = ["loma", "roma", "splg"]

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    h_pairs = filter_pairs_by_dataset(get_homography_benchmark_pairs(args.data_dir), args.dataset)
    cache = ensure_cached_matches(h_pairs, args.features, data_dir=args.data_dir, max_workers=min(12, args.workers))
    summary, curves = run_homography_benchmark(h_pairs, cache, args.features, max_workers=args.workers)
    print_markdown_summary(summary)
    if not args.no_plots:
        plot_homography_curves(curves, assets_dir)

    json_path = os.path.join(args.data_dir, "homography_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "curves": curves}, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
