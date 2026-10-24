"""Relative pose benchmark: Fundamental (F) and Essential (E) matrix estimation,
SupeRANSAC vs. PoseLib, on real detector matches with calibrated ground truth.

Datasets (frozen list in tests/datasets/pairs/calibrated_pairs.json):
    PhotoTourism (100 pairs), ETH3D (100 pairs), 7Scenes (100 pairs).

Protocol (identical for both methods):
    * input  : the raw matches of the detector, sorted by its confidence, same array for both
    * budget : min_iterations = 1000, max_iterations = 5000 (main table);
               min = max = b for b in BUDGETS (speed-accuracy sweep)
    * thresh : 1.5 px Sampson error (PoseLib normalises it by the focal length for E; so does SupeRANSAC)
    * PROSAC : enabled for both; PoseLib's max_prosac_iterations is set to max_iterations
    * metric : max(rotation error, translation angular error) from the pose recovered on the
               method's own inliers (cv2.recoverPose for F/E; PoseLib's pose for E) -> AUC@5/10/20 deg

Usage:
    python tests/test_relative_pose.py --problem all --features all
    python tests/test_relative_pose.py --problem fundamental --features LoMa --ablate_pp
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

from datasets.benchmark_pairs import (
    DEFAULT_DATA_DIR,
    canonical_detector_name,
    ensure_cached_matches,
    load_calibrated_pairs,
)
from evaluation import evaluate_R_t, pose_auc

cv2.setNumThreads(1)

BUDGETS = [25, 50, 100, 250, 500, 1000, 2500, 5000]
DEFAULT_FEATURES = ["loma", "roma", "splg"]
THRESHOLD_PX = 1.5
AUC_THRESHOLDS = [5.0, 10.0, 20.0]

FEATURE_FILE_TAG = {"LoMa": "loma", "RoMa": "roma", "DISK+LG": "splg", "LoFTR": "loftr",
                    "RootSIFT": "rootsift", "SIFT": "sift", "ORB": "orb"}
FEATURE_TITLE_TAG = {"LoMa": "LoMa", "RoMa": "RoMa", "DISK+LG": "DISK + LightGlue", "LoFTR": "LoFTR",
                     "RootSIFT": "RootSIFT", "SIFT": "SIFT", "ORB": "ORB"}

METHOD_COLORS = {"SupeRANSAC": "#1b9e77", "SupeRANSAC (no P&P)": "#7570b3", "PoseLib": "#377eb8"}
METHOD_STYLES = {"SupeRANSAC": "-o", "SupeRANSAC (no P&P)": "-.^", "PoseLib": "--s"}


# --------------------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------------------
def _pose_error_from_matrix(pair, corrs, M, inliers, is_fundamental):
    """Recover (R, t) from the estimated F or E using the method's own inliers; return max angular error."""
    if M is None or M.shape != (3, 3) or not np.all(np.isfinite(M)) or np.linalg.norm(M) < 1e-12:
        return 180.0
    inliers = np.asarray(inliers)
    if inliers.dtype == bool:
        inl_pts = corrs[inliers]
    else:
        inl_pts = corrs[inliers.astype(int)] if len(inliers) > 0 else np.empty((0, 4))
    if len(inl_pts) < (7 if is_fundamental else 5):
        return 180.0
    K1, K2 = pair["K1"], pair["K2"]
    E = K2.T @ M @ K1 if is_fundamental else M
    p1 = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
    p2 = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
    try:
        _, R, t, _ = cv2.recoverPose(E.astype(np.float64), p1, p2)
        er, et = evaluate_R_t(pair["R"], pair["t"], R, t)
        return float(max(er, et))
    except cv2.error:
        return 180.0


def _poselib_opts(max_iters, min_iters):
    return {
        "max_epipolar_error": THRESHOLD_PX,
        "min_iterations": min_iters,
        "max_iterations": max_iters,
        "max_prosac_iterations": max_iters,
        "progressive_sampling": True,
    }


POSELIB_BUNDLE = {"loss_type": "CAUCHY", "loss_scale": 0.5 * THRESHOLD_PX}


def _superansac_cfg(max_iters, min_iters, plane_parallax=None):
    """Default RANSACSettings with only the iteration budget set. plane_parallax=None keeps the library default."""
    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations = min_iters
    cfg.max_iterations = max_iters
    if plane_parallax is not None:
        cfg.fundamental_plane_parallax_post_refinement = plane_parallax
    return cfg


# --------------------------------------------------------------------------------------
# Per-pair evaluation (runs in worker processes)
# --------------------------------------------------------------------------------------
def _eval_f_pair(args):
    pair, corrs, scores, max_iters, min_iters, ablate_pp = args
    if len(corrs) < 8:
        return None
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    p1 = np.ascontiguousarray(corrs[:, :2])
    p2 = np.ascontiguousarray(corrs[:, 2:4])
    out = {"dataset": pair["dataset"], "scene": pair["scene"], "n": len(corrs)}

    default_pp = pysuperansac.RANSACSettings().fundamental_plane_parallax_post_refinement
    variants = [("SupeRANSAC", None)]
    if ablate_pp:
        variants.append((f"SupeRANSAC ({'no ' if default_pp else ''}P&P)", not default_pp))
    for name, pp in variants:
        cfg = _superansac_cfg(max_iters, min_iters, plane_parallax=pp)
        t0 = time.perf_counter()
        F, inl, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, probs, config=cfg)
        dt = (time.perf_counter() - t0) * 1000.0
        out[name] = (_pose_error_from_matrix(pair, corrs, F, inl, True), dt, len(inl))

    t0 = time.perf_counter()
    F, info = poselib.estimate_fundamental(p1, p2, _poselib_opts(max_iters, min_iters), POSELIB_BUNDLE)
    dt = (time.perf_counter() - t0) * 1000.0
    mask = np.asarray(info.get("inliers", []), dtype=bool)
    out["PoseLib"] = (_pose_error_from_matrix(pair, corrs, F, mask, True), dt, int(mask.sum()))
    return out


def _eval_e_pair(args):
    pair, corrs, scores, max_iters, min_iters, _ablate = args
    if len(corrs) < 8:
        return None
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    probs = np.ascontiguousarray(scores, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    K1 = np.ascontiguousarray(pair["K1"], dtype=np.float64)
    K2 = np.ascontiguousarray(pair["K2"], dtype=np.float64)
    p1 = np.ascontiguousarray(corrs[:, :2])
    p2 = np.ascontiguousarray(corrs[:, 2:4])
    cam1 = {"model": "PINHOLE", "width": int(pair["w1"]), "height": int(pair["h1"]),
            "params": [float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])]}
    cam2 = {"model": "PINHOLE", "width": int(pair["w2"]), "height": int(pair["h2"]),
            "params": [float(K2[0, 0]), float(K2[1, 1]), float(K2[0, 2]), float(K2[1, 2])]}
    out = {"dataset": pair["dataset"], "scene": pair["scene"], "n": len(corrs)}

    cfg = _superansac_cfg(max_iters, min_iters)
    t0 = time.perf_counter()
    E, inl, _, _ = pysuperansac.estimateEssentialMatrix(corrs, K1, K2, bbox, probs, config=cfg)
    dt = (time.perf_counter() - t0) * 1000.0
    out["SupeRANSAC"] = (_pose_error_from_matrix(pair, corrs, E, inl, False), dt, len(inl))

    t0 = time.perf_counter()
    pose, info = poselib.estimate_relative_pose(p1, p2, cam1, cam2, _poselib_opts(max_iters, min_iters), POSELIB_BUNDLE)
    dt = (time.perf_counter() - t0) * 1000.0
    n_inl = int(np.sum(info.get("inliers", [])))
    if pose is None or n_inl < 5 or not np.all(np.isfinite(pose.R)) or not np.all(np.isfinite(pose.t)):
        err = 180.0
    else:
        er, et = evaluate_R_t(pair["R"], pair["t"], pose.R, pose.t)
        err = float(max(er, et))
    out["PoseLib"] = (err, dt, n_inl)
    return out


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def summarize(results, methods):
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)
    by_ds["Overall"] = results
    summary = {}
    for ds_name, rs in by_ds.items():
        summary[ds_name] = {"num_pairs": len(rs)}
        for m in methods:
            errs = [r[m][0] for r in rs]
            ts = [r[m][1] for r in rs]
            inls = [r[m][2] for r in rs]
            summary[ds_name][m] = {
                "auc": [100.0 * a for a in pose_auc(errs, AUC_THRESHOLDS)],
                "med_err": float(np.median(errs)),
                "time_ms": float(np.mean(ts)),
                "med_ms": float(np.median(ts)),
                "inliers": float(np.mean(inls)),
            }
    return summary


def run_problem(problem, pairs, cache, features, ablate_pp, max_workers):
    eval_fn = _eval_f_pair if problem == "fundamental" else _eval_e_pair
    methods = ["SupeRANSAC"] + (["SupeRANSAC (no P&P)"] if (ablate_pp and problem == "fundamental") else []) + ["PoseLib"]
    summary, curves = {}, {}
    for feat in features:
        det = canonical_detector_name(feat)
        tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], 5000, 1000, ablate_pp) for p in pairs]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = [r for r in ex.map(eval_fn, tasks) if r is not None]
        summary[det] = summarize(results, methods)
        summary[det]["_methods"] = methods

        curves[det] = {m: [] for m in methods}
        for b in BUDGETS:
            b_tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], b, b, ablate_pp) for p in pairs]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                b_res = [r for r in ex.map(eval_fn, b_tasks) if r is not None]
            s = summarize(b_res, methods)["Overall"]
            for m in methods:
                curves[det][m].append({"budget": b, "time_ms": s[m]["time_ms"], "auc": s[m]["auc"]})
    return summary, curves


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def print_markdown(problem, summary):
    title = "Fundamental Matrix (F)" if problem == "fundamental" else "Essential Matrix (E)"
    print(f"\n### {title} -- real detector matches, calibrated GT\n")
    print("| Feature | Dataset | Pairs | Method | AUC@5° | AUC@10° | AUC@20° | Med err (°) | Mean ms | Med ms |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for det, ds_dict in summary.items():
        methods = ds_dict["_methods"]
        for ds_name, md in ds_dict.items():
            if ds_name == "_methods":
                continue
            best = [max(md[m]["auc"][i] for m in methods) for i in range(3)]
            for m in methods:
                st = md[m]
                cells = [f"**{v:.2f}%**" if abs(v - b) < 1e-9 else f"{v:.2f}%" for v, b in zip(st["auc"], best)]
                print(f"| {det} | {ds_name} | {md['num_pairs']} | {m} | {cells[0]} | {cells[1]} | {cells[2]} | "
                      f"{st['med_err']:.2f} | {st['time_ms']:.1f} | {st['med_ms']:.1f} |")


def plot_curves(problem, curves, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    prefix = "fundamental_matrix" if problem == "fundamental" else "essential_matrix"
    title = "Fundamental Matrix (F)" if problem == "fundamental" else "Essential Matrix (E)"
    for det, data in curves.items():
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0), dpi=180)
        fig.patch.set_facecolor("white")
        for m, pts in data.items():
            ax.plot([p["time_ms"] for p in pts], [p["auc"][1] for p in pts], METHOD_STYLES.get(m, "-o"),
                    color=METHOD_COLORS.get(m, "gray"), linewidth=2.5, markersize=7, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("Average runtime (ms, log scale)", fontsize=12)
        ax.set_ylabel("Pose AUC@10° (%)", fontsize=12)
        ax.set_title(f"{title} -- {FEATURE_TITLE_TAG.get(det, det)}", fontsize=13)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=True, fontsize=11)
        plt.tight_layout()
        out = os.path.join(assets_dir, f"{prefix}_{FEATURE_FILE_TAG.get(det, det.lower())}.png")
        plt.savefig(out, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser(description="Relative pose (F & E) benchmark: SupeRANSAC vs. PoseLib")
    parser.add_argument("--problem", choices=["all", "fundamental", "essential"], default="all")
    parser.add_argument("--dataset", choices=["all", "phototourism", "eth3d", "7scenes"], default="all")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--ablate_pp", action="store_true",
                        help="Also evaluate SupeRANSAC with the F Plane-and-Parallax post-refinement disabled")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    if "all" in [f.lower() for f in args.features]:
        args.features = list(DEFAULT_FEATURES)

    pairs = load_calibrated_pairs(args.data_dir)
    if args.dataset != "all":
        name = {"phototourism": "PhotoTourism", "eth3d": "ETH3D", "7scenes": "7Scenes"}[args.dataset]
        pairs = [p for p in pairs if p["dataset"] == name]
    cache = ensure_cached_matches(pairs, args.features, data_dir=args.data_dir, max_workers=min(12, args.workers))

    assets_dir = os.path.join(SUPERANSAC_DIR, "assets")
    json_path = os.path.join(args.data_dir, "relative_pose_results.json")
    out_json = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            out_json = json.load(f)

    for problem in ("fundamental", "essential"):
        if args.problem not in ("all", problem):
            continue
        summary, curves = run_problem(problem, pairs, cache, args.features, args.ablate_pp, args.workers)
        print_markdown(problem, summary)
        if not args.no_plots:
            plot_curves(problem, curves, assets_dir)
        out_json[problem] = {"summary": summary, "curves": curves, "num_pairs": len(pairs)}

    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=1)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
