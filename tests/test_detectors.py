"""Multi-detector benchmark: SupeRANSAC vs. PoseLib across 7 feature matchers
(LoMa, RoMa, LoFTR, DISK+LightGlue, RootSIFT, SIFT, ORB) on the same real pairs and protocol as
test_relative_pose.py (F, E: 300 calibrated pairs) and test_homography.py (H: Homogr/EVD[/HPatches]).

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

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERANSAC_DIR = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.benchmark_pairs import (
    ALL_DETECTORS,
    DEFAULT_DATA_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    load_calibrated_pairs,
    load_planar_pairs,
)
import test_relative_pose as trp
import test_homography as th

PROBLEMS = {
    "fundamental": (trp._eval_f_pair, trp.summarize, "Fundamental matrix (F)", "Pose AUC@10° (%)", "detector_fundamental.png"),
    "essential": (trp._eval_e_pair, trp.summarize, "Essential matrix (E)", "Pose AUC@10° (%)", "detector_essential.png"),
    "homography": (th._eval_h_pair, th.summarize, "Homography (H)", "Corner AUC@5px (%)", "detector_homography.png"),
}


def run(problem, pairs, cache, detectors, workers):
    eval_fn, summarize, *_ = PROBLEMS[problem]
    out = {}
    for det in detectors:
        if problem == "homography":
            tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], 5000, 1000) for p in pairs]
        else:
            tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], 5000, 1000, False) for p in pairs]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = [r for r in ex.map(eval_fn, tasks) if r is not None]
        s = summarize(results, ["SupeRANSAC", "PoseLib"]) if problem != "homography" else summarize(results)
        out[det] = s
    return out


def print_table(problem, res):
    _, _, title, metric, _ = PROBLEMS[problem]
    cols = ("AUC@5°", "AUC@10°", "AUC@20°") if problem != "homography" else ("AUC@2.5px", "AUC@5px", "AUC@10px")
    print(f"\n### {title} -- 7 matchers, SupeRANSAC vs. PoseLib\n")
    print(f"| Matcher | Pairs | PoseLib {cols[0]} / {cols[1]} / {cols[2]} | SupeRANSAC {cols[0]} / {cols[1]} / {cols[2]} | Δ {cols[1]} | Med err PoseLib / SupeRANSAC | ms PoseLib / SupeRANSAC |")
    print("|---|---:|---|---|---:|---|---|")
    su_all, pl_all = [], []
    for det, s in res.items():
        o = s["Overall"]
        su, pl = o["SupeRANSAC"], o["PoseLib"]
        su_all.append(su["auc"]); pl_all.append(pl["auc"])
        d = su["auc"][1] - pl["auc"][1]
        print(f"| {det} | {o['num_pairs']} | {pl['auc'][0]:.1f} / {pl['auc'][1]:.1f} / {pl['auc'][2]:.1f} | "
              f"{su['auc'][0]:.1f} / {su['auc'][1]:.1f} / {su['auc'][2]:.1f} | {d:+.2f} | "
              f"{pl['med_err']:.2f} / {su['med_err']:.2f} | {pl['time_ms']:.0f} / {su['time_ms']:.0f} |")
    su_m, pl_m = np.mean(su_all, axis=0), np.mean(pl_all, axis=0)
    print(f"| **Average** | | {pl_m[0]:.1f} / {pl_m[1]:.1f} / {pl_m[2]:.1f} | {su_m[0]:.1f} / {su_m[1]:.1f} / {su_m[2]:.1f} | {su_m[1]-pl_m[1]:+.2f} | | |")


def plot(problem, res, assets_dir):
    _, _, title, metric, fname = PROBLEMS[problem]
    dets = list(res.keys())
    x = np.arange(len(dets))
    w = 0.38
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=180)
    fig.patch.set_facecolor("white")
    su = [res[d]["Overall"]["SupeRANSAC"]["auc"][1] for d in dets]
    pl = [res[d]["Overall"]["PoseLib"]["auc"][1] for d in dets]
    for off, vals, name, col in ((-w / 2, su, "SupeRANSAC", "#1b9e77"), (w / 2, pl, "PoseLib", "#377eb8")):
        bars = ax.bar(x + off, vals, w, label=name, color=col, edgecolor="black", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dets, fontsize=10)
    ax.set_ylabel(metric)
    ax.set_title(f"{title} -- {res[dets[0]]['Overall']['num_pairs']} real pairs", fontsize=12)
    ax.set_ylim(0, max(max(su), max(pl)) * 1.2 + 2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(assets_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Multi-detector evaluation (7 matchers)")
    parser.add_argument("--problem", choices=["all", "fundamental", "essential", "homography"], default="all")
    parser.add_argument("--detectors", "--features", dest="detectors", nargs="+", default=ALL_DETECTORS)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    detectors = list(ALL_DETECTORS) if "all" in [d.lower() for d in args.detectors] else [canonical_detector_name(d) for d in args.detectors]

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    json_path = os.path.join(args.data_dir, "detector_results.json")
    out_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            out_json = json.load(f)
    cal_pairs = planar_pairs = None
    for problem in ("fundamental", "essential", "homography"):
        if args.problem not in ("all", problem):
            continue
        if problem == "homography":
            planar_pairs = planar_pairs or load_planar_pairs(args.data_dir)
            pairs = planar_pairs
        else:
            cal_pairs = cal_pairs or load_calibrated_pairs(args.data_dir)
            pairs = cal_pairs
        cache = ensure_cached_matches(pairs, detectors, data_dir=args.data_dir, max_workers=min(16, args.workers))
        res = run(problem, pairs, cache, detectors, args.workers)
        print_table(problem, res)
        if not args.no_plots:
            plot(problem, res, assets_dir)
        out_json[problem] = res

    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=1)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
