"""Results tables, generated only from the JSON outputs in data/ (so numbers are never typed by hand).

    python tests/make_readme_tables.py           # writes RESULTS.md (all detailed tables) and prints the README summary
    python tests/make_readme_tables.py summary   # prints only the README summary table

Confidence intervals: 95% paired cluster bootstrap over image pairs / scenes (see make_readme_figures.auc_ci).
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)

import numpy as np

from benchmark_protocol import SWEEP_MIN_ITERS, TABLE_MAX_ITERS, TABLE_MIN_ITERS
from evaluation import pose_auc
from make_readme_figures import AUC_T, N_BOOT, auc_ci, load

LEARNED = ("LoMa", "RoMa", "LoFTR", "DISK+LG")
CLASSICAL = ("RootSIFT", "SIFT", "ORB")
ALL7 = LEARNED + CLASSICAL


def b(v, best):
    return f"**{v:.1f}**" if abs(v - best) < 1e-9 else f"{v:.1f}"


def ci_str(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def group_delta_ci(blocks, kind, seed=0):
    """Mean over matchers of AUC(SupeRANSAC) - AUC(PoseLib), with a cluster bootstrap that resamples the
    image pairs jointly for all matchers (the matchers are evaluated on the same pairs)."""
    ths, j = AUC_T[kind]
    auc = lambda e: 100.0 * pose_auc(e, ths)[j]
    keys = sorted(set().union(*[set(bl["keys"]) for bl in blocks]))
    kid = {k: i for i, k in enumerate(keys)}
    per = []
    for bl in blocks:
        lists = [[] for _ in keys]
        for i, k in enumerate(bl["keys"]):
            lists[kid[k]].append(i)
        per.append(lists)
    su = [np.asarray(bl["errors"]["SupeRANSAC"]) for bl in blocks]
    pl = [np.asarray(bl["errors"]["PoseLib"]) for bl in blocks]
    point = np.mean([auc(s) - auc(p) for s, p in zip(su, pl)])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        sample = rng.integers(0, len(keys), len(keys))
        ds = []
        for s, p, lists in zip(su, pl, per):
            sel = np.fromiter((i for k in sample for i in lists[k]), dtype=int)
            if len(sel):
                ds.append(auc(s[sel]) - auc(p[sel]))
        boot.append(np.mean(ds))
    return (point, *np.percentile(boot, [2.5, 97.5]))


# ------------------------------------------------------------------------------------------------
# README summary
# ------------------------------------------------------------------------------------------------
def summary_table(det, cls, p6d):
    rows = []
    for prob, lab, unit, data in (("fundamental", "$F$", "AUC@10°", "300 calibrated pairs"),
                                  ("essential", "$E$", "AUC@10°", "300 calibrated pairs"),
                                  ("homography", "$H$", "AUC@5px", "16 Homogr + EVD pairs")):
        for grp, ms in (("learned matchers", LEARNED), ("classical matchers", CLASSICAL)):
            os_ = [det[prob][m]["Overall"] for m in ms]
            pl = np.mean([o["PoseLib"]["auc"][1] for o in os_]); su = np.mean([o["SupeRANSAC"]["auc"][1] for o in os_])
            tpl = np.mean([o["PoseLib"]["time_ms"] for o in os_]); tsu = np.mean([o["SupeRANSAC"]["time_ms"] for o in os_])
            d = group_delta_ci([o["per_pair"] for o in os_], prob)
            rows.append(f"| {lab}, {grp} ({', '.join(ms)}) | {data} | {unit} | {pl:.1f} | {su:.1f} | {ci_str(d)} | {tpl:.0f} / {tsu:.0f} |")
    for key, lab, unit, data, kind in (("fundamental", "$F$, labelled SIFT tentative matches", "AUC@2px", "30 scenes × 5", "classic_f"),
                                       ("homography", "$H$, labelled SIFT tentative matches", "AUC@5px", "35 scenes × 5", "classic_h")):
        o = cls[key]["Overall"]
        d = auc_ci(o["per_pair"], ["SupeRANSAC", "PoseLib"], kind)["delta"]
        rows.append(f"| {lab} | {data} | {unit} | {o['PoseLib']['auc'][1]:.1f} | {o['SupeRANSAC']['auc'][1]:.1f} | {ci_str(d)} | "
                    f"{o['PoseLib']['time_ms']:.0f} / {o['SupeRANSAC']['time_ms']:.0f} |")
    pn = p6d["absolute_pose"]["by_dataset"]["7Scenes RGB-D"]
    d = auc_ci(pn["per_pair"], ["SupeRANSAC", "PoseLib"], "6d")["delta"]
    rows.append(f"| PnP, 6 matchers | {pn['num_problems']} 7Scenes RGB-D problems | AUC@(5°,10cm) | {pn['PoseLib']['auc'][1]:.1f} | "
                f"{pn['SupeRANSAC']['auc'][1]:.1f} | {ci_str(d)} | {pn['PoseLib']['time_ms']:.0f} / {pn['SupeRANSAC']['time_ms']:.0f} |")
    rg = p6d["rigid_transform"]["by_dataset"]["7Scenes RGB-D"]
    d = auc_ci(rg["per_pair"], ["SupeRANSAC", "Procrustes RANSAC"], "6d")["delta"]
    rows.append(f"| Rigid, 6 matchers (baseline: Procrustes RANSAC) | {rg['num_problems']} 7Scenes RGB-D problems | AUC@(5°,10cm) | "
                f"{rg['Procrustes RANSAC']['auc'][1]:.1f} | {rg['SupeRANSAC']['auc'][1]:.1f} | {ci_str(d)} | Python / {rg['SupeRANSAC']['time_ms']:.0f} |")
    head = ["| Task | Data | Metric | PoseLib | SupeRANSAC | Δ [95% CI] | ms PoseLib / SupeRANSAC |",
            "|---|---|---|---:|---:|---:|---:|"]
    return "\n".join(head + rows)


# ------------------------------------------------------------------------------------------------
# RESULTS.md tables
# ------------------------------------------------------------------------------------------------
def rel_table(rel, problem, idx=1):
    s = rel[problem]["summary"]
    out = ["| Matcher | Dataset | Pairs | PoseLib | SupeRANSAC | Δ | ms PoseLib / SupeRANSAC |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in ("LoMa", "RoMa", "DISK+LG"):
        for ds in ("PhotoTourism", "ETH3D", "7Scenes", "Overall"):
            d = s[m][ds]; pl, su = d["PoseLib"]["auc"][idx], d["SupeRANSAC"]["auc"][idx]
            best = max(pl, su)
            name = "**Overall**" if ds == "Overall" else ds
            out.append(f"| {m} | {name} | {d['num_pairs']} | {b(pl, best)} | {b(su, best)} | {su-pl:+.1f} | {d['PoseLib']['time_ms']:.0f} / {d['SupeRANSAC']['time_ms']:.0f} |")
    return "\n".join(out)


def hom_table(hom, idx=1):
    s = hom["summary"]
    out = ["| Matcher | Dataset | Pairs | PoseLib | SupeRANSAC | Δ | ms PoseLib / SupeRANSAC |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in ("LoMa", "RoMa", "DISK+LG"):
        for ds in [k for k in s[m] if k != "Overall"] + ["Overall"]:
            d = s[m][ds]; pl, su = d["PoseLib"]["auc"][idx], d["SupeRANSAC"]["auc"][idx]
            best = max(pl, su)
            name = "**Overall**" if ds == "Overall" else ds
            out.append(f"| {m} | {name} | {d['num_pairs']} | {b(pl, best)} | {b(su, best)} | {su-pl:+.1f} | {d['PoseLib']['time_ms']:.0f} / {d['SupeRANSAC']['time_ms']:.0f} |")
    return "\n".join(out)


def det_table(det):
    cols = (("fundamental", "$F$ AUC@10°"), ("essential", "$E$ AUC@10°"), ("homography", "$H$ AUC@5px"))
    out = ["| Matcher | " + " | ".join(f"{c} PoseLib / SupeRANSAC | Δ [95% CI] | ms PoseLib / SupeRANSAC" for _, c in cols) + " |",
           "|---|" + "---:|---:|---:|" * len(cols)]
    for m in ALL7:
        cells = []
        for p, _ in cols:
            o = det[p][m]["Overall"]; pl, su = o["PoseLib"]["auc"][1], o["SupeRANSAC"]["auc"][1]
            d = auc_ci(o["per_pair"], ["SupeRANSAC", "PoseLib"], p)["delta"]
            cells.append(f"{b(pl, max(pl, su))} / {b(su, max(pl, su))} | {ci_str(d)} | {o['PoseLib']['time_ms']:.0f} / {o['SupeRANSAC']['time_ms']:.0f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def sweep_table(rel, p6d):
    out = [f"| Problem | Matcher | Method | " + " | ".join(f"min {m}" for m in SWEEP_MIN_ITERS) + " |",
           "|---|---|---|" + "---:|" * len(SWEEP_MIN_ITERS)]
    for prob, lab in (("fundamental", "$F$"), ("essential", "$E$")):
        for m, curves in rel[prob]["curves"].items():
            for meth in ("PoseLib", "SupeRANSAC"):
                pts = curves[meth]
                out.append(f"| {lab} | {m} | {meth} | " + " | ".join(f"{p['auc'][1]:.1f} ({p['time_ms']:.0f} ms)" for p in pts) + " |")
    for meth in ("PoseLib", "SupeRANSAC"):
        pts = p6d["absolute_pose"]["curves"][meth]
        out.append(f"| PnP | 6 matchers | {meth} | " + " | ".join(f"{p['auc'][1]:.1f} ({p['time_ms']:.0f} ms)" for p in pts) + " |")
    cv = p6d["absolute_pose"]["curves"]["OpenCV RANSAC"]
    out.append("")
    out.append("OpenCV RANSAC (PnP, no minimum iteration count; iteration cap): " +
               ", ".join(f"cap {p['max_iterations']}: {p['auc'][1]:.1f} ({p['time_ms']:.0f} ms)" for p in cv) + ".")
    return "\n".join(out)


def cls_table(cls):
    out = ["| Problem | Scenes × repeats | Method | AUC@1px / 2px / 5px ($F$) — AUC@2.5 / 5 / 10 px ($H$) | Med err (px) | Mean ms |", "|---|---:|---|---|---:|---:|"]
    for p, lab in (("fundamental", "$F$"), ("homography", "$H$")):
        s = cls[p]
        for ds in [k for k in s if k != "Overall"] + ["Overall"]:
            d = s[ds]; name = f"{lab} **Overall**" if ds == "Overall" else f"{lab} {ds}"
            best = [max(d["PoseLib"]["auc"][i], d["SupeRANSAC"]["auc"][i]) for i in range(3)]
            for m in ("PoseLib", "SupeRANSAC"):
                a = d[m]["auc"]
                out.append(f"| {name} | {d['num_problems']} × {d['num_runs'] // d['num_problems']} | {m} | {b(a[0], best[0])} / {b(a[1], best[1])} / {b(a[2], best[2])} | {d[m]['med_err']:.2f} | {d[m]['time_ms']:.0f} |")
    return "\n".join(out)


def p6d_table(p6d):
    out = ["| Problem | Data | Method | AUC@(2°,4cm) | AUC@(5°,10cm) | AUC@(10°,20cm) | Med rot / trans | Mean ms |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, lab, methods in (("absolute_pose", "PnP", ("SupeRANSAC", "PoseLib", "OpenCV RANSAC")), ("rigid_transform", "Rigid", ("SupeRANSAC", "Procrustes RANSAC"))):
        s = p6d[key]["by_dataset"]
        for ds in [k for k in s if k != "Overall"]:
            d = s[ds]; best = [max(d[m]["auc"][i] for m in methods) for i in range(3)]
            name = ds.replace(" (synthetic outliers)", " synthetic") + (" synthetic" if ds == "rigid_pose_example" else "")
            for m in methods:
                a = d[m]["auc"]
                ms = "Python" if m == "Procrustes RANSAC" else f"{d[m]['time_ms']:.0f}"
                out.append(f"| {lab} | {name}, {d['num_problems']} | {m} | {b(a[0], best[0])} | {b(a[1], best[1])} | {b(a[2], best[2])} | {d[m]['med_rot']:.2f}° / {d[m]['med_trans']:.1f} cm | {ms} |")
    return "\n".join(out)


def p6d_matcher_table(p6d):
    out = ["| Matcher | PnP: PoseLib / SupeRANSAC / OpenCV | Δ vs PoseLib [95% CI] | Rigid: Procrustes / SupeRANSAC | Δ [95% CI] |", "|---|---:|---:|---:|---:|"]
    pn, rg = p6d["absolute_pose"]["by_detector"], p6d["rigid_transform"]["by_detector"]
    for m in pn:
        a = pn[m]; r = rg.get(m)
        dp = auc_ci(a["per_pair"], ["SupeRANSAC", "PoseLib"], "6d")["delta"]
        cells = f"| {m} | {a['PoseLib']['auc'][1]:.1f} / {a['SupeRANSAC']['auc'][1]:.1f} / {a['OpenCV RANSAC']['auc'][1]:.1f} | {ci_str(dp)} | "
        if r is not None:
            dr = auc_ci(r["per_pair"], ["SupeRANSAC", "Procrustes RANSAC"], "6d")["delta"]
            cells += f"{r['Procrustes RANSAC']['auc'][1]:.1f} / {r['SupeRANSAC']['auc'][1]:.1f} | {ci_str(dr)} |"
        else:
            cells += "– | – |"
        out.append(cells)
    return "\n".join(out)


RESULTS_HEADER = f"""# SupeRANSAC vs. PoseLib — detailed results

Generated by `tests/make_readme_tables.py` from the JSON outputs of the benchmark scripts in `tests/` (do not edit by hand).
The README shows the summary and three figures; this file has every table. Protocol: see the README. Briefly, every method
runs its own adaptive termination within {TABLE_MIN_ITERS}–{TABLE_MAX_ITERS} iterations on identical inputs and thresholds;
Δ = SupeRANSAC − baseline in AUC points; [95% CI] = paired cluster bootstrap over image pairs / scenes. Runtimes are means
per problem, single-threaded calls, measured with up to 8 concurrent benchmark processes.
"""


def write_results_md(rel, hom, det, cls, p6d, path):
    parts = [RESULTS_HEADER,
             "## 1. Relative pose and homography on real detector matches\n",
             "**Data.** 300 calibrated image pairs frozen in `tests/datasets/pairs/calibrated_pairs.json`: PhotoTourism *palace_of_westminster* "
             "(100, CVPR'20 RANSAC tutorial GT), ETH3D (100: courtyard 20, facade 20, electro / kicker / relief 15, playground 10, terrace 5; "
             "COLMAP GT; images downscaled 4×), 7Scenes (100: chess 40, fire 40, heads 20; dataset poses). Homography: the 16 Homogr + EVD "
             "pairs with ground-truth $H$.\n",
             "**Metric.** $F$, $E$: relative pose recovered from the estimated matrix on the method's own inliers (`cv2.recoverPose`; PoseLib's "
             "own pose for $E$), error = max(rotation, translation angle), AUC@5/10/20°. $H$: mean transfer error of the corners of the "
             "ground-truth-inlier bounding box, AUC@2.5/5/10 px.\n",
             "### Fundamental matrix (AUC@10°, 1.5 px)\n", rel_table(rel, "fundamental"), "",
             "### Essential matrix (AUC@10°, 1.5 px)\n", rel_table(rel, "essential"), "",
             "### Homography (AUC@5px, 4.0 px)\n", hom_table(hom), "",
             "## 2. Seven feature matchers\n",
             "Same pairs with **LoMa**, **RoMa**, **LoFTR**, **DISK + LightGlue**, **RootSIFT**, **SIFT** and **ORB**. The classical matchers "
             "use a 0.95 ratio test on 4000 keypoints (typically 10–40 % inliers); the learned matchers deliver mostly clean sets.\n",
             det_table(det), "",
             "## 3. Speed–accuracy sweep\n",
             f"`min_iterations` ∈ {SWEEP_MIN_ITERS} with each library's adaptive termination and `max_iterations = {TABLE_MAX_ITERS}`; "
             f"the `min {TABLE_MIN_ITERS}` column is the table configuration. Cells: AUC (mean ms).\n",
             sweep_table(rel, p6d), "",
             "## 4. Labelled tentative correspondences (high outlier ratios)\n",
             "The original SIFT tentative matches shipped with **kusvod2**, **AdelaideRMF / Multi-H** ($F$; 30 scenes, 5–94 % inliers, "
             "median 48 %) and **Homogr**, **EVD**, **AdelaideRMF / Multi-H planes** ($H$; 35 scenes, 2–49 % inliers, median 22 %). No "
             "confidence scores exist and the files are sorted by label, so every repeat shuffles them with a fixed seed and both methods "
             "use uniform sampling; 5 repeats per scene. $F$ error = mean Sampson error on the labelled inliers; $H$ error = corner "
             "transfer error against the dataset homography.\n",
             cls_table(cls), "",
             "## 5. 6D absolute pose (PnP) and rigid registration\n",
             "**Real data.** The 100 frozen 7Scenes pairs × 6 matchers (LoMa, RoMa, DISK+LG, RootSIFT, SIFT, ORB); matches lifted through the "
             "depth maps; GT = dataset poses; outliers = the matcher's own mismatches and depth noise. Error = max(rotation error, "
             "0.5 × translation error in cm). **Baselines.** PnP: PoseLib `estimate_absolute_pose` and `cv2.solvePnPRansac` (EPnP). Rigid: "
             "a 3-point Procrustes RANSAC (MSAC, adaptive termination, final least squares; `tests/test_6d_pose.py`), written in Python, "
             "so its runtime is not comparable. Matches ordered by confidence are sampled with PROSAC by every method. **Synthetic stress "
             "tests:** `pose6dscene` (33–91 % random 2D–3D outliers) and `rigid_pose_example` (70–92 % random 3D–3D outliers).\n",
             p6d_table(p6d), "", "Per matcher on 7Scenes (AUC@(5°,10cm)):\n", p6d_matcher_table(p6d), ""]
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("Wrote", path)


if __name__ == "__main__":
    rel, hom, det, cls, p6d = (load(n) for n in ("relative_pose_results.json", "homography_results.json", "detector_results.json",
                                                   "classic_labelled_results.json", "pose_6d_results.json"))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        write_results_md(rel, hom, det, cls, p6d, os.path.join(ROOT, "RESULTS.md"))
    print("\n<!-- summary -->\n" + summary_table(det, cls, p6d) + "\n")
