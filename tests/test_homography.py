"""Homography benchmark: SupeRANSAC vs. PoseLib on real detector matches with ground-truth homographies.

Datasets: Homogr + EVD (GT H from the dataset), HPatches viewpoint sequences if present.

Protocol (identical for both methods):
    * input  : raw detector matches sorted by confidence (same array for both)
    * budget : each method's own adaptive termination within 1000-5000 iterations (no sweep: 16 pairs are
               too few for a stable speed-accuracy curve, one pair moves the AUC by up to 6 points)
    * thresh : 4.0 px reprojection error for both (SupeRANSAC's default for H; PoseLib max_reproj_error)
    * metric : mean transfer error of the 4 corners of the GT-inlier bounding box under H_est vs. H_gt
               -> AUC@2.5/5/10 px

Usage:
    python tests/test_homography.py --features all
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERANSAC_DIR = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import cv2
import numpy as np
import poselib
import pysuperansac

from datasets.benchmark_pairs import (
    DEFAULT_DATA_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    load_planar_pairs,
)
from evaluation import pose_auc
from benchmark_protocol import TABLE_MAX_ITERS, TABLE_MIN_ITERS, default_workers, per_problem_record

cv2.setNumThreads(1)

DEFAULT_FEATURES = ["loma", "roma", "splg"]
THRESHOLD_PX = 4.0
AUC_THRESHOLDS = [2.5, 5.0, 10.0]


def corner_error(pair, H_est, n_inliers):
    """Mean transfer error of the corners of the GT-inlier bounding box (image 1) under H_est vs. H_gt."""
    if n_inliers < 4 or H_est is None or H_est.shape != (3, 3) or not np.all(np.isfinite(H_est)) or np.linalg.norm(H_est) < 1e-12:
        return 1e6
    gt = pair["gt_inliers"]
    mn, mx = np.min(gt[:, :2], axis=0), np.max(gt[:, :2], axis=0)
    if np.any(mx - mn < 20.0):
        mn = np.array([0.2 * pair["w1"], 0.2 * pair["h1"]])
        mx = np.array([0.8 * pair["w1"], 0.8 * pair["h1"]])
    corners = np.array([[mn[0], mn[1], 1], [mx[0], mn[1], 1], [mx[0], mx[1], 1], [mn[0], mx[1], 1]], dtype=np.float64)
    c_gt = corners @ pair["H_gt"].T
    c_est = corners @ H_est.T
    if np.any(np.abs(c_gt[:, 2]) < 1e-8) or np.any(np.abs(c_est[:, 2]) < 1e-8):
        return 1e6
    c_gt = c_gt[:, :2] / c_gt[:, 2:3]
    c_est = c_est[:, :2] / c_est[:, 2:3]
    return float(np.mean(np.linalg.norm(c_est - c_gt, axis=1)))


def _eval_h_pair(args):
    pair, corrs, scores, max_iters, min_iters = args
    if len(corrs) < 4:
        return None
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    p1 = np.ascontiguousarray(corrs[:, :2])
    p2 = np.ascontiguousarray(corrs[:, 2:4])
    out = {"dataset": pair["dataset"], "scene": pair["scene"], "n": len(corrs)}

    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations = min_iters
    cfg.max_iterations = max_iters
    t0 = time.perf_counter()
    H, inl, _, _ = pysuperansac.estimateHomography(corrs, bbox, probs, config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    out["SupeRANSAC"] = (corner_error(pair, H, len(inl)), dt, len(inl))

    ropt = {"max_reproj_error": THRESHOLD_PX, "min_iterations": min_iters, "max_iterations": max_iters,
            "max_prosac_iterations": max_iters, "progressive_sampling": True}
    bopt = {"loss_type": "CAUCHY", "loss_scale": 0.5 * THRESHOLD_PX}
    t0 = time.perf_counter()
    H, info = poselib.estimate_homography(p1, p2, ropt, bopt)
    dt = (time.perf_counter() - t0) * 1000.0
    n_inl = int(np.sum(info.get("inliers", [])))
    out["PoseLib"] = (corner_error(pair, H, n_inl), dt, n_inl)
    return out


def summarize(results, methods=("SupeRANSAC", "PoseLib")):
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)
    by_ds["Overall"] = results
    summary = {}
    for ds_name, rs in by_ds.items():
        summary[ds_name] = {"num_pairs": len(rs)}
        for m in methods:
            errs = [r[m][0] for r in rs]
            summary[ds_name][m] = {
                "auc": [100.0 * a for a in pose_auc(errs, AUC_THRESHOLDS)],
                "med_err": float(np.median(errs)),
                "time_ms": float(np.mean([r[m][1] for r in rs])),
                "med_ms": float(np.median([r[m][1] for r in rs])),
                "inliers": float(np.mean([r[m][2] for r in rs])),
            }
    summary["Overall"]["per_pair"] = per_problem_record(results, methods)
    return summary


def run_benchmark(pairs, cache, features, max_workers):
    summary = {}
    for feat in features:
        det = canonical_detector_name(feat)
        tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], TABLE_MAX_ITERS, TABLE_MIN_ITERS) for p in pairs]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = [r for r in ex.map(_eval_h_pair, tasks) if r is not None]
        summary[det] = summarize(results)
    return summary


def print_markdown(summary):
    print("\n### Homography (H) -- real detector matches, GT homographies\n")
    print("| Feature | Dataset | Pairs | Method | AUC@2.5px | AUC@5px | AUC@10px | Med err (px) | Mean ms | Med ms |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for det, ds_dict in summary.items():
        for ds_name, md in ds_dict.items():
            best = [max(md[m]["auc"][i] for m in ("SupeRANSAC", "PoseLib")) for i in range(3)]
            for m in ("SupeRANSAC", "PoseLib"):
                st = md[m]
                cells = [f"**{v:.2f}%**" if abs(v - b) < 1e-9 else f"{v:.2f}%" for v, b in zip(st["auc"], best)]
                print(f"| {det} | {ds_name} | {md['num_pairs']} | {m} | {cells[0]} | {cells[1]} | {cells[2]} | "
                      f"{st['med_err']:.2f} | {st['time_ms']:.1f} | {st['med_ms']:.1f} |")


def main():
    parser = argparse.ArgumentParser(description="Homography benchmark: SupeRANSAC vs. PoseLib")
    parser.add_argument("--dataset", choices=["all", "homogr", "evd", "hpatches"], default="all")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    if "all" in [f.lower() for f in args.features]:
        args.features = list(DEFAULT_FEATURES)

    pairs = load_planar_pairs(args.data_dir)
    if args.dataset != "all":
        name = {"homogr": "Homogr", "evd": "EVD", "hpatches": "HPatches"}[args.dataset]
        pairs = [p for p in pairs if p["dataset"] == name]
    cache = ensure_cached_matches(pairs, args.features, data_dir=args.data_dir, max_workers=min(12, args.workers))

    summary = run_benchmark(pairs, cache, args.features, args.workers)
    print_markdown(summary)

    json_path = os.path.join(args.data_dir, "homography_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "num_pairs": len(pairs)}, f, indent=1)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
