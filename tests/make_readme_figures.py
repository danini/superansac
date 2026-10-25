"""README figures, drawn only from the JSON outputs in data/ (so figures, tables and JSON always agree).

    Fig. 1  assets/fig_accuracy_runtime.png  -- F / E / H, 7 matchers, default settings: PoseLib -> SupeRANSAC
    Fig. 2  assets/fig_speed_accuracy.png    -- F / E / PnP speed-accuracy sweeps (min_iterations), default = star
    Fig. 3  assets/fig_hard_and_6d.png       -- labelled tentative matches (F, H) and 6D (PnP, rigid) on 7Scenes

Uncertainty: 95% paired cluster-bootstrap confidence intervals (resampling image pairs / scenes) -- error bars in
Fig. 3, and a '*' after the ΔAUC label in Fig. 1 when the interval of the difference excludes zero.

Usage: python tests/make_readme_figures.py
"""
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_protocol import TABLE_MAX_ITERS, TABLE_MIN_ITERS
from evaluation import pose_auc

DATA = os.path.join(ROOT, "data")
ASSETS = os.path.join(ROOT, "assets")
MATCHERS = ["LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"]
MATCHER_COLORS = dict(zip(MATCHERS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]))
METHOD_COLORS = {"SupeRANSAC": "#1b9e77", "PoseLib": "#377eb8", "OpenCV RANSAC": "#e41a1c", "Procrustes RANSAC": "#999999"}
AUC_T = {"fundamental": ([5.0, 10.0, 20.0], 1), "essential": ([5.0, 10.0, 20.0], 1), "homography": ([2.5, 5.0, 10.0], 1),
         "classic_f": ([1.0, 2.0, 5.0], 1), "classic_h": ([2.5, 5.0, 10.0], 1), "6d": ([2.0, 5.0, 10.0], 1)}
N_BOOT = 2000


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def _clusters(keys):
    idx = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, []).append(i)
    return list(idx.values())


def auc_ci(per_pair, methods, kind, seed=0):
    """AUC (%) of every method and of the difference methods[0] - methods[1], with 95% cluster-bootstrap CIs."""
    ths, j = AUC_T[kind]
    errs = {m: np.asarray(per_pair["errors"][m]) for m in methods}
    auc = lambda e: 100.0 * pose_auc(e, ths)[j]
    point = {m: auc(errs[m]) for m in methods}
    clusters = _clusters(per_pair["keys"])
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in methods}
    for _ in range(N_BOOT):
        sel = np.concatenate([clusters[c] for c in rng.integers(0, len(clusters), len(clusters))])
        for m in methods:
            boot[m].append(auc(errs[m][sel]))
    out = {m: (point[m], *np.percentile(boot[m], [2.5, 97.5])) for m in methods}
    if len(methods) >= 2:
        d = np.asarray(boot[methods[0]]) - np.asarray(boot[methods[1]])
        out["delta"] = (point[methods[0]] - point[methods[1]], *np.percentile(d, [2.5, 97.5]))
    return out


def fig_accuracy_runtime(det, out_path):
    titles = {"fundamental": "Fundamental matrix  (AUC@10°)", "essential": "Essential matrix  (AUC@10°)",
              "homography": "Homography  (AUC@5px, 16 pairs)"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    for ax, prob in zip(axes, ("fundamental", "essential", "homography")):
        pts = []
        for m in MATCHERS:
            o = det[prob][m]["Overall"]
            ci = auc_ci(o["per_pair"], ["SupeRANSAC", "PoseLib"], prob)
            pts.append((m, o["PoseLib"]["time_ms"], ci["PoseLib"][0], o["SupeRANSAC"]["time_ms"], ci["SupeRANSAC"][0], ci["delta"]))
        xs = [p[1] for p in pts] + [p[3] for p in pts]; ys = [p[2] for p in pts] + [p[4] for p in pts]
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 1.9, max(xs) * 1.35)
        ax.set_ylim(min(ys) - 3, max(ys) + 3)
        placed = []  # label anchors in points, to avoid overlapping labels
        to_pt = 72.0 / fig.dpi
        for m, tp, ap, ts, a_s, (d, dl, dh) in sorted(pts, key=lambda p: -p[4]):
            c = MATCHER_COLORS[m]
            ax.annotate("", xy=(ts, a_s), xytext=(tp, ap), arrowprops=dict(arrowstyle="->", color=c, lw=1.5, alpha=0.85))
            ax.plot([tp], [ap], "s", mfc="white", mec=c, ms=8, mew=1.8)
            ax.plot([ts], [a_s], "o", color=c, ms=8)
            sig = "*" if (dl > 0 or dh < 0) else ""
            left = ts <= tp  # put the label on the side away from the arrow
            x, y = (v * to_pt for v in ax.transData.transform((ts, a_s)))
            dy = 0.0
            while any(abs(px - x) < 75 and abs(py - (y + dy)) < 10 for px, py in placed):
                dy -= 10
            placed.append((x, y + dy))
            ax.annotate(f"{m} {d:+.1f}{sig}", xy=(ts, a_s), xytext=(-8 if left else 8, dy), textcoords="offset points",
                        fontsize=8.5, color=c, ha="right" if left else "left", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.set_xlabel("mean runtime per problem (ms, log scale)")
        ax.set_ylabel("AUC (%)")
        ax.set_title(titles[prob], fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    handles = [plt.Line2D([], [], marker="s", ls="", mfc="white", mec="black", ms=8, label="PoseLib"),
               plt.Line2D([], [], marker="o", ls="", color="black", ms=8, label="SupeRANSAC, labelled with ΔAUC (* = 95% CI excludes 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Real matches from 7 matchers at the default settings: arrows point from PoseLib to SupeRANSAC "
                 "(up = more accurate, left = faster)", fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def fig_speed_accuracy(rel, p6d, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    for ax, prob, title in ((axes[0], "fundamental", "Fundamental matrix"), (axes[1], "essential", "Essential matrix")):
        for m, curves in rel[prob]["curves"].items():
            c = MATCHER_COLORS[m]
            for meth, style, mk in (("SupeRANSAC", "-", "o"), ("PoseLib", "--", "s")):
                pts = curves[meth]
                ax.plot([p["time_ms"] for p in pts], [p["auc"][1] for p in pts], style, marker=mk, color=c, ms=4.5, lw=1.8,
                        mfc=c if meth == "SupeRANSAC" else "white")
                star = [p for p in pts if p["min_iterations"] == TABLE_MIN_ITERS]
                ax.plot([star[0]["time_ms"]], [star[0]["auc"][1]], "*", color=c, ms=15, mec="black", mew=0.8)
            last = curves["SupeRANSAC"][-1]
            ax.annotate(m, xy=(last["time_ms"], last["auc"][1]), xytext=(6, 0), textcoords="offset points", fontsize=9, color=c, va="center")
        ax.set_title(f"{title}  (AUC@10°)", fontsize=12)
    ax = axes[2]
    for meth, style, mk in (("SupeRANSAC", "-", "o"), ("PoseLib", "--", "s"), ("OpenCV RANSAC", ":", "d")):
        pts = p6d["absolute_pose"]["curves"][meth]
        c = METHOD_COLORS[meth]
        ax.plot([p["time_ms"] for p in pts], [p["auc"][1] for p in pts], style, marker=mk, color=c, ms=4.5, lw=1.8, label=meth)
        star = [p for p in pts if p.get("min_iterations", TABLE_MIN_ITERS) == TABLE_MIN_ITERS and p.get("max_iterations", TABLE_MAX_ITERS) == TABLE_MAX_ITERS]
        ax.plot([star[0]["time_ms"]], [star[0]["auc"][1]], "*", color=c, ms=15, mec="black", mew=0.8)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("PnP, 7Scenes RGB-D, 6 matchers  (AUC@5°,10cm)", fontsize=12)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("mean runtime per problem (ms, log scale)")
        ax.set_ylabel("AUC (%)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    handles = [plt.Line2D([], [], ls="-", marker="o", color="black", ms=5, label="SupeRANSAC"),
               plt.Line2D([], [], ls="--", marker="s", color="black", mfc="white", ms=5, label="PoseLib"),
               plt.Line2D([], [], ls="", marker="*", color="gray", mec="black", ms=13, label="default settings (the tables)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Speed-accuracy trade-off: min_iterations swept from 0 to {TABLE_MAX_ITERS} with each library's adaptive "
                 f"termination (max {TABLE_MAX_ITERS}); OpenCV: iteration cap", fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def fig_hard_and_6d(cls, p6d, out_path):
    groups = [
        ("F, labelled tentative\nmatches (AUC@2px)", cls["fundamental"]["Overall"], ["SupeRANSAC", "PoseLib"], "classic_f"),
        ("H, labelled tentative\nmatches (AUC@5px)", cls["homography"]["Overall"], ["SupeRANSAC", "PoseLib"], "classic_h"),
        ("PnP, 7Scenes RGB-D\n(AUC@5°,10cm)", p6d["absolute_pose"]["by_dataset"]["7Scenes RGB-D"], ["SupeRANSAC", "PoseLib", "OpenCV RANSAC"], "6d"),
        ("Rigid, 7Scenes RGB-D\n(AUC@5°,10cm)", p6d["rigid_transform"]["by_dataset"]["7Scenes RGB-D"], ["SupeRANSAC", "Procrustes RANSAC"], "6d"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8), dpi=160)
    w = 0.26
    for gi, (label, summ, methods, kind) in enumerate(groups):
        ci = auc_ci(summ["per_pair"], methods, kind)
        for mi, m in enumerate(methods):
            x = gi + (mi - (len(methods) - 1) / 2) * w
            a, lo, hi = ci[m]
            ax.bar(x, a, w * 0.92, color=METHOD_COLORS[m], edgecolor="black", lw=0.5,
                   label=m if gi == (2 if m == "OpenCV RANSAC" else 3 if m == "Procrustes RANSAC" else 0) else None)
            ax.errorbar([x], [a], yerr=[[a - lo], [hi - a]], fmt="none", ecolor="black", capsize=3, lw=1)
            ms = summ[m]["time_ms"]
            txt = f"{a:.1f}\n{ms:.0f} ms" if m != "Procrustes RANSAC" else f"{a:.1f}\n(Python)"
            ax.text(x, hi + 1.0, txt, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("AUC (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title("Low inlier ratios (labelled SIFT tentative matches, uniform sampling) and 6D pose; 95% CIs", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    det, rel, cls, p6d = load("detector_results.json"), load("relative_pose_results.json"), load("classic_labelled_results.json"), load("pose_6d_results.json")
    fig_accuracy_runtime(det, os.path.join(ASSETS, "fig_accuracy_runtime.png"))
    fig_speed_accuracy(rel, p6d, os.path.join(ASSETS, "fig_speed_accuracy.png"))
    fig_hard_and_6d(cls, p6d, os.path.join(ASSETS, "fig_hard_and_6d.png"))


if __name__ == "__main__":
    main()

