"""Runnable benchmark script for Multi-Detector Evaluation across 7 Feature Matchers (Section 3).

Evaluates SupeRANSAC vs. PoseLib across 7 feature detectors/matchers:
1. LoMa (ECCV 2026 Oral - DaD + DeDoDe-G + LoMa-B)
2. RoMa (Dense Foundation Matcher)
3. LoFTR (Detector-Free Transformer Matcher)
4. DISK + LightGlue (Learned Keypoints + Graph Attention Matcher)
5. RootSIFT (Hellinger-Normalized SIFT + SNN Ratio)
6. SIFT (Standard SIFT + SNN Ratio)
7. ORB (Binary Descriptor + Hamming SNN Ratio)

Across all 3 two-view geometry tasks (Fundamental F, Essential E, Homography H)
and all benchmark datasets (PhotoTourism, ETH3D, 7Scenes, Kusvod2/AdelaideRMF, Homogr/EVD).

Usage:
    python tests/test_detectors.py --problem all
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERANSAC_DIR = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.multi_benchmark import (
    ALL_DETECTORS,
    DEFAULT_DATA_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    get_essential_benchmark_pairs,
    get_fundamental_benchmark_pairs,
    get_homography_benchmark_pairs,
)
from evaluation import pose_auc
from test_homography import _eval_h_pair, prepare_homography_gt
from test_relative_pose import _eval_e_pair, _eval_f_pair


def run_multi_detector_eval(problem, pairs, cache, detectors, max_workers=24):
    if problem == "fundamental":
        eval_fn = _eval_f_pair
        auc_thresholds = [5.0, 10.0, 20.0]
        valid_pairs = pairs
    elif problem == "essential":
        eval_fn = _eval_e_pair
        auc_thresholds = [5.0, 10.0, 20.0]
        valid_pairs = pairs
    else:
        eval_fn = _eval_h_pair
        auc_thresholds = [2.5, 5.0, 10.0]
        valid_pairs = [hp for hp in (prepare_homography_gt(p, cache) for p in pairs) if hp is not None]

    n_runs = 1
    out = {}
    for det in detectors:
        det_key = canonical_detector_name(det)
        tasks = [
            (p, cache[p["cache_key"]][det_key][0], cache[p["cache_key"]][det_key][1], 5000, 1000, n_runs)
            for p in valid_pairs
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = [r for r in ex.map(eval_fn, tasks) if r is not None]

        su_e = [e for r in results for e in r["su_errs"]]
        pl_e = [e for r in results for e in r["pl_errs"]]
        su_t = [t for r in results for t in r["su_ts"]]
        pl_t = [t for r in results for t in r["pl_ts"]]
        su_i = [i for r in results for i in r["su_inls"]]
        pl_i = [i for r in results for i in r["pl_inls"]]

        su_aucs = [x * 100.0 for x in pose_auc(su_e, auc_thresholds)]
        pl_aucs = [x * 100.0 for x in pose_auc(pl_e, auc_thresholds)]

        out[det_key] = {
            "num_pairs": len(results),
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
    return out


def plot_detector_chart(problem, res_dict, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    detectors = list(res_dict.keys())
    x = np.arange(len(detectors))
    w = 0.36

    if problem == "fundamental":
        title = "Fundamental Matrix (F) Across 7 Feature Matchers"
        m_labels = ["AUC@5° (%)", "AUC@10° (%)", "AUC@20° (%)"]
        fname = "detector_fundamental.png"
    elif problem == "essential":
        title = "Essential Matrix (E) Across 7 Feature Matchers"
        m_labels = ["AUC@5° (%)", "AUC@10° (%)", "AUC@20° (%)"]
        fname = "detector_essential.png"
    else:
        title = "Homography (H) Across 7 Feature Matchers"
        m_labels = ["AUC@2.5 px (%)", "AUC@5.0 px (%)", "AUC@10.0 px (%)"]
        fname = "detector_homography.png"

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), dpi=180)
    fig.patch.set_facecolor("white")

    for idx_m in range(3):
        ax = axes[idx_m]
        su_vals = [res_dict[d]["SupeRANSAC"]["auc"][idx_m] for d in detectors]
        pl_vals = [res_dict[d]["PoseLib"]["auc"][idx_m] for d in detectors]

        b1 = ax.bar(x - w / 2, su_vals, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + w / 2, pl_vals, w, label="PoseLib", color="#377eb8", edgecolor="black", linewidth=0.6)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0f6b4e")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#235784")

        ax.set_xticks(x)
        ax.set_xticklabels(detectors, rotation=28, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylabel(m_labels[idx_m], fontsize=10, fontweight="bold")
        ax.set_title(m_labels[idx_m].split(" ")[0], fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(max(su_vals), max(pl_vals)) * 1.18 + 3.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx_m == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    # Panel 4: Runtime comparison
    ax_t = axes[3]
    su_t = [res_dict[d]["SupeRANSAC"]["time_ms"] for d in detectors]
    pl_t = [res_dict[d]["PoseLib"]["time_ms"] for d in detectors]
    b1 = ax_t.bar(x - w / 2, su_t, w, label="SupeRANSAC (Ours)", color="#1b9e77", edgecolor="black", linewidth=0.6)
    b2 = ax_t.bar(x + w / 2, pl_t, w, label="PoseLib", color="#377eb8", edgecolor="black", linewidth=0.6)
    for bar in b1:
        h = bar.get_height()
        ax_t.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0f6b4e")
    for bar in b2:
        h = bar.get_height()
        ax_t.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#235784")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(detectors, rotation=28, ha="right", fontsize=9, fontweight="bold")
    ax_t.set_ylabel("Mean Runtime (ms)", fontsize=10, fontweight="bold")
    ax_t.set_title("Runtime (ms) ↓", fontsize=11, fontweight="bold")
    ax_t.set_ylim(0, max(max(su_t), max(pl_t)) * 1.22 + 1.0)
    ax_t.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)

    fig.suptitle(f"{title} (Multi-Dataset Benchmark)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(assets_dir, fname)
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def print_detector_table(problem, res_dict):
    if problem == "fundamental":
        title = "Fundamental Matrix (F)"
        cols = ("AUC@5°", "AUC@10°", "AUC@20°")
    elif problem == "essential":
        title = "Essential Matrix (E)"
        cols = ("AUC@5°", "AUC@10°", "AUC@20°")
    else:
        title = "Homography (H)"
        cols = ("AUC@2.5px", "AUC@5.0px", "AUC@10.0px")

    print(f"\n### {title} — 7 Feature Matchers\n")
    print(f"| Detector / Matcher | Method | {cols[0]} ↑ | {cols[1]} ↑ | {cols[2]} ↑ | Med Error ↓ | Mean ms ↓ | Med ms ↓ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for det_key, m_dict in res_dict.items():
        for m_name in ["SupeRANSAC", "PoseLib"]:
            st = m_dict[m_name]
            a0, a1, a2 = st["auc"]
            b = "**" if m_name == "SupeRANSAC" else ""
            print(
                f"| {det_key} | {b}{m_name}{b} | "
                f"{b}{a0:.2f}%{b} | {b}{a1:.2f}%{b} | {b}{a2:.2f}%{b} | "
                f"{st['med_err']:.2f} | {st['time_ms']:.2f} | {st['med_ms']:.2f} |"
            )


def main():
    parser = argparse.ArgumentParser(description="Multi-Detector Evaluation Across 7 Feature Matchers")
    parser.add_argument("--problem", choices=["all", "fundamental", "essential", "homography"], default="all")
    parser.add_argument("--detectors", nargs="+", default=ALL_DETECTORS)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    out_json = {}

    if args.problem in ("all", "fundamental"):
        f_pairs = get_fundamental_benchmark_pairs(args.data_dir)
        cache = ensure_cached_matches(f_pairs, args.detectors, data_dir=args.data_dir, max_workers=min(16, args.workers))
        f_res = run_multi_detector_eval("fundamental", f_pairs, cache, args.detectors, max_workers=args.workers)
        print_detector_table("fundamental", f_res)
        if not args.no_plots:
            plot_detector_chart("fundamental", f_res, assets_dir)
        out_json["fundamental"] = f_res

    if args.problem in ("all", "essential"):
        e_pairs = get_essential_benchmark_pairs(args.data_dir)
        cache = ensure_cached_matches(e_pairs, args.detectors, data_dir=args.data_dir, max_workers=min(16, args.workers))
        e_res = run_multi_detector_eval("essential", e_pairs, cache, args.detectors, max_workers=args.workers)
        print_detector_table("essential", e_res)
        if not args.no_plots:
            plot_detector_chart("essential", e_res, assets_dir)
        out_json["essential"] = e_res

    if args.problem in ("all", "homography"):
        h_pairs = get_homography_benchmark_pairs(args.data_dir)
        cache = ensure_cached_matches(h_pairs, args.detectors, data_dir=args.data_dir, max_workers=min(16, args.workers))
        h_res = run_multi_detector_eval("homography", h_pairs, cache, args.detectors, max_workers=args.workers)
        print_detector_table("homography", h_res)
        if not args.no_plots:
            plot_detector_chart("homography", h_res, assets_dir)
        out_json["homography"] = h_res

    json_path = os.path.join(args.data_dir, "detector_results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()

