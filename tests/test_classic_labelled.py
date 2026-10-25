"""Classic labelled-correspondence benchmark (high outlier ratios): SupeRANSAC vs. PoseLib.

Input = the ORIGINAL tentative correspondences shipped with the datasets (real SIFT matches,
real outliers, real multi-structure scenes). No detector, no injected points, no synthetic scores.

    F : kusvod2 + AdelaideRMF/Multi-H  (GT = labelled inliers)
        metric: mean Sampson error (px) of the estimated F on the labelled inliers -> AUC@1/2/5 px
    H : Homogr + EVD (GT H from the dataset) + AdelaideRMF/Multi-H largest labelled plane
        metric: corner transfer error vs. GT H -> AUC@2.5/5/10 px

The files are stored sorted by label, and no confidence scores exist, so every repeat shuffles the
rows with a fixed seed and BOTH methods use uniform sampling (PROSAC off). Thresholds: 1.5 px (F),
4.0 px (H). Budget: 1000-5000 iterations for both. 5 repeats per problem.

Usage:
    python tests/test_classic_labelled.py --problem all
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

from datasets.benchmark_pairs import (
    DEFAULT_DATA_DIR,
    load_classic_f_problems,
    load_classic_h_problems,
    sampson_errors,
)
from evaluation import pose_auc
from test_homography import corner_error

cv2.setNumThreads(1)

N_REPEATS = 5
F_THRESHOLD, H_THRESHOLD = 1.5, 4.0
F_AUC, H_AUC = [1.0, 2.0, 5.0], [2.5, 5.0, 10.0]


def _shuffle(prob, rep):
    rng = np.random.default_rng(1000 * rep + 17)
    perm = rng.permutation(len(prob["corrs"]))
    return np.ascontiguousarray(prob["corrs"][perm], dtype=np.float64)


def _eval_f(args):
    prob, rep, max_iters, min_iters = args
    corrs = _shuffle(prob, rep)
    bbox = np.array([prob["h1"], prob["w1"], prob["h2"], prob["w2"]], dtype=np.float64)
    gt = prob["gt_inliers"]
    out = {"dataset": prob["dataset"], "scene": prob["scene"], "rep": rep}

    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations, cfg.max_iterations = min_iters, max_iters
    cfg.sampler = pysuperansac.SamplerType.Uniform
    t0 = time.perf_counter()
    F, inl, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, [], config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    err = float(np.mean(sampson_errors(gt, F))) if (F is not None and len(inl) >= 7 and np.all(np.isfinite(F))) else 1e6
    out["SupeRANSAC"] = (err, dt, len(inl))

    ropt = {"max_epipolar_error": F_THRESHOLD, "min_iterations": min_iters, "max_iterations": max_iters,
            "progressive_sampling": False}
    bopt = {"loss_type": "CAUCHY", "loss_scale": 0.5 * F_THRESHOLD}
    t0 = time.perf_counter()
    F, info = poselib.estimate_fundamental(np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4]), ropt, bopt)
    dt = (time.perf_counter() - t0) * 1000.0
    n_inl = int(np.sum(info.get("inliers", [])))
    err = float(np.mean(sampson_errors(gt, F))) if (F is not None and n_inl >= 7 and np.all(np.isfinite(F))) else 1e6
    out["PoseLib"] = (err, dt, n_inl)
    return out


def _eval_h(args):
    prob, rep, max_iters, min_iters = args
    corrs = _shuffle(prob, rep)
    bbox = np.array([prob["h1"], prob["w1"], prob["h2"], prob["w2"]], dtype=np.float64)
    out = {"dataset": prob["dataset"], "scene": prob["scene"], "rep": rep}

    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations, cfg.max_iterations = min_iters, max_iters
    cfg.sampler = pysuperansac.SamplerType.Uniform
    t0 = time.perf_counter()
    H, inl, _, _ = pysuperansac.estimateHomography(corrs, bbox, [], config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    out["SupeRANSAC"] = (corner_error(prob, H, len(inl)), dt, len(inl))

    ropt = {"max_reproj_error": H_THRESHOLD, "min_iterations": min_iters, "max_iterations": max_iters,
            "progressive_sampling": False}
    bopt = {"loss_type": "CAUCHY", "loss_scale": 0.5 * H_THRESHOLD}
    t0 = time.perf_counter()
    H, info = poselib.estimate_homography(np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4]), ropt, bopt)
    dt = (time.perf_counter() - t0) * 1000.0
    n_inl = int(np.sum(info.get("inliers", [])))
    out["PoseLib"] = (corner_error(prob, H, n_inl), dt, n_inl)
    return out


def summarize(results, aucs):
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)
    by_ds["Overall"] = results
    summary = {}
    for ds, rs in by_ds.items():
        summary[ds] = {"num_runs": len(rs), "num_problems": len({r["scene"] for r in rs})}
        for m in ("SupeRANSAC", "PoseLib"):
            errs = [r[m][0] for r in rs]
            summary[ds][m] = {
                "auc": [100.0 * a for a in pose_auc(errs, aucs)],
                "med_err": float(np.median(errs)),
                "time_ms": float(np.mean([r[m][1] for r in rs])),
                "inliers": float(np.mean([r[m][2] for r in rs])),
            }
    return summary


def print_markdown(title, summary, cols, unit):
    print(f"\n### {title}\n")
    print(f"| Dataset | Problems x repeats | Method | {cols[0]} | {cols[1]} | {cols[2]} | Med err ({unit}) | Mean ms |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for ds, md in summary.items():
        best = [max(md[m]["auc"][i] for m in ("SupeRANSAC", "PoseLib")) for i in range(3)]
        for m in ("SupeRANSAC", "PoseLib"):
            st = md[m]
            cells = [f"**{v:.2f}%**" if abs(v - b) < 1e-9 else f"{v:.2f}%" for v, b in zip(st["auc"], best)]
            print(f"| {ds} | {md['num_problems']} x {md['num_runs'] // max(1, md['num_problems'])} | {m} | "
                  f"{cells[0]} | {cells[1]} | {cells[2]} | {st['med_err']:.2f} | {st['time_ms']:.1f} |")


def plot_summary(f_summary, h_summary, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, summary, title, ylabel, idx in (
        (axes[0], f_summary, "Fundamental matrix -- kusvod2 / AdelaideRMF / Multi-H", "AUC@2px (%)", 1),
        (axes[1], h_summary, "Homography -- Homogr / EVD / AdelaideRMF planes", "AUC@5px (%)", 1),
    ):
        if summary is None:
            ax.axis("off")
            continue
        names = [k for k in summary if k != "Overall"] + ["Overall"]
        x = np.arange(len(names))
        w = 0.36
        su = [summary[n]["SupeRANSAC"]["auc"][idx] for n in names]
        pl = [summary[n]["PoseLib"]["auc"][idx] for n in names]
        b1 = ax.bar(x - w / 2, su, w, label="SupeRANSAC", color="#1b9e77", edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x + w / 2, pl, w, label="PoseLib", color="#377eb8", edgecolor="black", linewidth=0.5)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.1f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = os.path.join(assets_dir, "classic_labelled.png")
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Classic labelled-correspondence benchmark (F, H)")
    parser.add_argument("--problem", choices=["all", "fundamental", "homography"], default="all")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    # Merge into existing results so that running a single problem keeps the other one.
    json_path = os.path.join(args.data_dir, "classic_labelled_results.json")
    out_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            out_json = json.load(f)
    f_summary = h_summary = None
    if args.problem in ("all", "fundamental"):
        probs = load_classic_f_problems(args.data_dir)
        tasks = [(p, r, 5000, 1000) for p in probs for r in range(N_REPEATS)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_eval_f, tasks))
        f_summary = summarize(res, F_AUC)
        print_markdown("Fundamental matrix -- classic labelled tentative matches", f_summary,
                       ("AUC@1px", "AUC@2px", "AUC@5px"), "px")
        out_json["fundamental"] = f_summary
    if args.problem in ("all", "homography"):
        probs = load_classic_h_problems(args.data_dir)
        tasks = [(p, r, 5000, 1000) for p in probs for r in range(N_REPEATS)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_eval_h, tasks))
        h_summary = summarize(res, H_AUC)
        print_markdown("Homography -- classic labelled tentative matches", h_summary,
                       ("AUC@2.5px", "AUC@5px", "AUC@10px"), "px")
        out_json["homography"] = h_summary
    if not args.no_plots:
        plot_summary(out_json.get("fundamental"), out_json.get("homography"), os.path.join(SUPERANSAC_DIR, "assets"))

    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=1)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
