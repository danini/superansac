"""Development-split ablation harness (tuning only; never used for reported numbers).

Runs SupeRANSAC with one or more sets of RANSACSettings overrides against PoseLib on the DEV split
(tests/datasets/pairs/calibrated_pairs_dev.json; Track-B tentative matches for H; dev 7Scenes pairs for PnP)
and prints a delta matrix.

Usage:
    python tests/dev_ablate.py --problems F E --features LoMa splg sift \\
        --config base '{}' --config no_degensac '{"degensac_settings.enabled": false}'
    python tests/dev_ablate.py --configs configs.json        # {"name": {"dotted.key": value, ...}, ...}

Override keys are dotted attribute paths on pysuperansac.RANSACSettings, e.g.
  "final_optimization_settings.max_iterations", "degensac_settings.enabled", "scoring" (string enum name).
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
    load_calibrated_pairs,
    load_classic_h_problems,
    load_planar_pairs,
)
from datasets.pose6d_problems import PNP_DETECTORS, load_7scenes_pnp_problems, load_7scenes_rigid_problems
from evaluation import pose_auc
import test_relative_pose as trp
import test_homography as th
import test_6d_pose as t6

cv2.setNumThreads(1)

ENUMS = {
    "scoring": pysuperansac.ScoringType,
    "sampler": pysuperansac.SamplerType,
    "local_optimization": pysuperansac.LocalOptimizationType,
    "final_optimization": pysuperansac.LocalOptimizationType,
    "neighborhood": pysuperansac.NeighborhoodType,
}


def make_cfg(overrides, max_iters=5000, min_iters=1000):
    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations, cfg.max_iterations = min_iters, max_iters
    for key, val in overrides.items():
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        if leaf in ENUMS and isinstance(val, str):
            val = getattr(ENUMS[leaf], val)
        if not hasattr(obj, leaf):
            raise AttributeError(f"RANSACSettings has no '{key}'")
        setattr(obj, leaf, val)
    return cfg


# ---------------------------------------------------------------- per-problem workers
def _f(args):
    pair, corrs, scores, overrides = args
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    out = {}
    for name, ov in overrides.items():
        t0 = time.perf_counter()
        try:
            F, inl, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, np.ascontiguousarray(scores), config=make_cfg(ov))
            out[name] = (trp._pose_error_from_matrix(pair, corrs, F, inl, True), (time.perf_counter() - t0) * 1e3)
        except RuntimeError as e:
            out[name] = (180.0, (time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter()
    F, info = poselib.estimate_fundamental(np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4]),
                                           trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
    out["PoseLib"] = (trp._pose_error_from_matrix(pair, corrs, F, np.asarray(info["inliers"], bool), True), (time.perf_counter() - t0) * 1e3)
    return pair["dataset"], out


def _e(args):
    pair, corrs, scores, overrides = args
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    K1, K2 = np.ascontiguousarray(pair["K1"]), np.ascontiguousarray(pair["K2"])
    out = {}
    for name, ov in overrides.items():
        t0 = time.perf_counter()
        try:
            E, inl, _, _ = pysuperansac.estimateEssentialMatrix(corrs, K1, K2, bbox, np.ascontiguousarray(scores), config=make_cfg(ov))
            out[name] = (trp._pose_error_from_matrix(pair, corrs, E, inl, False), (time.perf_counter() - t0) * 1e3)
        except RuntimeError:
            out[name] = (180.0, (time.perf_counter() - t0) * 1e3)
    cam = lambda K, w, h: {"model": "PINHOLE", "width": int(w), "height": int(h),
                           "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}
    t0 = time.perf_counter()
    pose, info = poselib.estimate_relative_pose(np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4]),
                                                cam(K1, pair["w1"], pair["h1"]), cam(K2, pair["w2"], pair["h2"]),
                                                trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
    dt = (time.perf_counter() - t0) * 1e3
    if pose is None or int(np.sum(info["inliers"])) < 5:
        out["PoseLib"] = (180.0, dt)
    else:
        er, et = trp.evaluate_R_t(pair["R"], pair["t"], pose.R, pose.t)
        out["PoseLib"] = (float(max(er, et)), dt)
    return pair["dataset"], out


def _h(args):
    pair, corrs, scores, overrides, uniform = args
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    out = {}
    for name, ov in overrides.items():
        cfg = make_cfg(ov)
        if uniform:
            cfg.sampler = pysuperansac.SamplerType.Uniform
        t0 = time.perf_counter()
        try:
            H, inl, _, _ = pysuperansac.estimateHomography(corrs, bbox, [] if uniform else np.ascontiguousarray(scores), config=cfg)
            out[name] = (th.corner_error(pair, H, len(inl)), (time.perf_counter() - t0) * 1e3)
        except RuntimeError:
            out[name] = (1e6, (time.perf_counter() - t0) * 1e3)
    ropt = {"max_reproj_error": th.THRESHOLD_PX, "min_iterations": 1000, "max_iterations": 5000,
            "max_prosac_iterations": 5000, "progressive_sampling": not uniform}
    t0 = time.perf_counter()
    H, info = poselib.estimate_homography(np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4]), ropt,
                                          {"loss_type": "CAUCHY", "loss_scale": 0.5 * th.THRESHOLD_PX})
    out["PoseLib"] = (th.corner_error(pair, H, int(np.sum(info["inliers"]))), (time.perf_counter() - t0) * 1e3)
    return pair["dataset"], out


def _pnp(args):
    prob, overrides = args
    corrs, K, thr, cm = prob["corrs"], prob["K"], prob["threshold_px"], prob["cm_scale"]
    pts2d, pts3d = np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:5])
    min_c = pts3d.min(axis=0)
    corrs_su = np.ascontiguousarray(np.hstack([pts2d, pts3d - min_c]))
    bbox = list(np.max(corrs_su, axis=0))
    out = {}
    for name, ov in overrides.items():
        cfg = make_cfg(ov)
        cfg.inlier_threshold = thr
        t0 = time.perf_counter()
        R, t, inl, _, _ = pysuperansac.estimateAbsolutePose(corrs_su, pysuperansac.CameraType.SimplePinhole,
                                                            [float(K[0, 0]), float(K[0, 2]), float(K[1, 2])], bbox,
                                                            prob["probs"] if len(prob["probs"]) else [], config=cfg)
        dt = (time.perf_counter() - t0) * 1e3
        if R is None or len(inl) < 4:
            out[name] = (180.0, dt)
        else:
            out[name] = (t6.pose_error(prob["R_gt"], prob["t_gt"], R, np.asarray(t).flatten() - R @ min_c, cm)[0], dt)
    cam = {"model": "PINHOLE", "width": int(prob["width"]), "height": int(prob["height"]),
           "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}
    ropt = {"max_reproj_error": thr, "min_iterations": 1000, "max_iterations": 5000}
    if len(prob["probs"]):  # confidence-ordered matches: PROSAC, as SupeRANSAC
        ropt.update({"progressive_sampling": True, "max_prosac_iterations": 5000})
    t0 = time.perf_counter()
    pose, info = poselib.estimate_absolute_pose(pts2d, pts3d, cam, ropt, {"loss_type": "CAUCHY", "loss_scale": 0.5 * thr})
    dt = (time.perf_counter() - t0) * 1e3
    out["PoseLib"] = ((180.0, dt) if pose is None else (t6.pose_error(prob["R_gt"], prob["t_gt"], pose.R, pose.t, cm)[0], dt))
    return prob["detector"], out


def _rigid(args):
    prob, overrides = args
    corrs, thr, cm = prob["corrs"], prob["threshold_3d"], prob["cm_scale"]
    mn = corrs.min(axis=0)
    shifted = np.ascontiguousarray(corrs - mn)
    bbox = np.ascontiguousarray(shifted.max(axis=0) + 1e-3)
    out = {}
    for name, ov in overrides.items():
        cfg = make_cfg(ov)
        cfg.inlier_threshold = thr
        t0 = time.perf_counter()
        T, inl, _, _ = pysuperansac.estimateRigidTransform(shifted, bbox, prob["probs"] if len(prob["probs"]) else [], config=cfg)
        dt = (time.perf_counter() - t0) * 1e3
        if T is None or len(inl) < 3 or not np.all(np.isfinite(T)):
            out[name] = (180.0, dt)
        else:
            T1 = np.eye(4); T1[3, :3] = -mn[:3]
            T2 = np.eye(4); T2[3, :3] = mn[3:]
            M = (T1 @ T @ T2).T
            out[name] = (t6.pose_error(prob["R_gt"], prob["t_gt"], M[:3, :3], M[:3, 3], cm)[0], dt)
    t0 = time.perf_counter()
    R, t, _ = t6.procrustes_ransac(corrs, thr, 5000, 1000, ordered=len(prob["probs"]) > 0)
    dt = (time.perf_counter() - t0) * 1e3
    out["PoseLib"] = ((180.0, dt) if R is None else (t6.pose_error(prob["R_gt"], prob["t_gt"], R, t, cm)[0], dt))  # column = Procrustes RANSAC baseline
    return prob["detector"], out


# ---------------------------------------------------------------- driver
def _bootstrap_delta(err_a, err_b, aucs, idx, n_boot=1000, seed=0):
    """Paired bootstrap of AUC(err_a) - AUC(err_b): returns (delta, lo95, hi95)."""
    err_a, err_b = np.asarray(err_a), np.asarray(err_b)
    n = len(err_a)
    rng = np.random.default_rng(seed)
    d0 = 100 * (pose_auc(err_a, aucs)[idx] - pose_auc(err_b, aucs)[idx])
    ds = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        ds[i] = 100 * (pose_auc(err_a[s], aucs)[idx] - pose_auc(err_b[s], aucs)[idx])
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def report(title, results, names, aucs, idx, fail_thr=None):
    """results: list of (group, {name: (err, ms)}).

    Per group: AUC (delta vs PoseLib) and mean runtime. Then, over ALL pairs, paired statistics for each
    config: bootstrap 95% CI of the AUC delta vs PoseLib and vs the first config, and win/loss counts
    (W = this config succeeds (< fail_thr) where the other fails (>= fail_thr), L = the reverse).
    """
    fail_thr = fail_thr if fail_thr is not None else aucs[-1]
    groups = sorted({g for g, _ in results})
    cols = names + ["PoseLib"]
    print(f"\n{title}  (metric: AUC@{aucs[idx]}; n={len(results)})")
    head = f"{'group':22s}" + "".join(f"{c[:18]:>20s}" for c in cols) + "   " + "".join(f"{'ms ' + c[:8]:>12s}" for c in cols)
    print(head)
    for g in groups + ["ALL"]:
        rs = [r for gg, r in results if g == "ALL" or gg == g]
        aucv = {c: 100 * pose_auc([r[c][0] for r in rs], aucs)[idx] for c in cols}
        ms = {c: np.mean([r[c][1] for r in rs]) for c in cols}
        pl = aucv["PoseLib"]
        line = f"{g:22s}" + "".join(f"{aucv[c]:8.2f} ({aucv[c]-pl:+5.2f})" if c != "PoseLib" else f"{aucv[c]:20.2f}" for c in cols)
        line += "   " + "".join(f"{ms[c]:12.1f}" for c in cols)
        print(line)
    errs = {c: np.array([r[c][0] for _, r in results]) for c in cols}
    base = names[0]
    print(f"  paired (fail >= {fail_thr}):")
    for c in names:
        d, lo, hi = _bootstrap_delta(errs[c], errs["PoseLib"], aucs, idx)
        w = int(np.sum((errs[c] < fail_thr) & (errs["PoseLib"] >= fail_thr)))
        l = int(np.sum((errs[c] >= fail_thr) & (errs["PoseLib"] < fail_thr)))
        line = f"    {c[:20]:20s} vs PoseLib {d:+5.2f} [{lo:+5.2f},{hi:+5.2f}] W/L {w:3d}/{l:3d}"
        if c != base:
            d, lo, hi = _bootstrap_delta(errs[c], errs[base], aucs, idx)
            w = int(np.sum((errs[c] < fail_thr) & (errs[base] >= fail_thr)))
            l = int(np.sum((errs[c] >= fail_thr) & (errs[base] < fail_thr)))
            line += f"   | vs {base[:12]} {d:+5.2f} [{lo:+5.2f},{hi:+5.2f}] W/L {w:3d}/{l:3d}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["F", "E", "H", "PnP"], choices=["F", "E", "H", "PnP", "Rigid"])
    ap.add_argument("--features", nargs="+", default=["LoMa", "splg", "sift"])
    ap.add_argument("--config", nargs=2, action="append", metavar=("NAME", "JSON"), default=[])
    ap.add_argument("--configs", type=str, default=None, help="JSON file {name: overrides}")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--save", type=str, default=None, help="Pickle all per-pair results to this path")
    args = ap.parse_args()
    saved = {}

    overrides = {}
    if args.configs:
        overrides.update(json.load(open(args.configs)))
    for name, js in args.config:
        overrides[name] = json.loads(js)
    if not overrides:
        overrides = {"base": {}}
    names = list(overrides.keys())
    print("configs:", json.dumps(overrides))
    if args.split == "test":
        print("!!! Running on the TEST split -- this counts against the evaluation budget !!!")

    feats = [canonical_detector_name(f) for f in args.features]
    if any(p in args.problems for p in ("F", "E")):
        pairs = load_calibrated_pairs(DEFAULT_DATA_DIR, split=args.split)
        cache = ensure_cached_matches(pairs, feats, data_dir=DEFAULT_DATA_DIR, max_workers=16)
        for prob, fn, aucs in (("F", _f, [5.0, 10.0, 20.0]), ("E", _e, [5.0, 10.0, 20.0])):
            if prob not in args.problems:
                continue
            for det in feats:
                tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], overrides) for p in pairs if len(cache[p["cache_key"]][det][0]) >= 8]
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    res = list(ex.map(fn, tasks))
                saved[(prob, det)] = [(t[0]["scene"], g, r) for t, (g, r) in zip(tasks, res)]
                report(f"[{prob}] {det}", res, names, aucs, 1)

    if "H" in args.problems:
        # dev: Track-B tentative matches (uniform sampling, 3 shuffles); also the 16 detector-match pairs (PROSAC)
        probs = load_classic_h_problems(DEFAULT_DATA_DIR, verbose=False)
        tasks = []
        for rep in range(3):
            for p in probs:
                rng = np.random.default_rng(1000 * rep + 17)
                perm = rng.permutation(len(p["corrs"]))
                tasks.append((p, p["corrs"][perm], p["scores"][perm], overrides, True))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_h, tasks))
        saved[("H", "classic")] = [(t[0]["scene"], g, r) for t, (g, r) in zip(tasks, res)]
        report("[H] classic tentative matches (uniform)", res, names, [2.5, 5.0, 10.0], 1)
        planar = load_planar_pairs(DEFAULT_DATA_DIR, verbose=False)
        cache = ensure_cached_matches(planar, feats, data_dir=DEFAULT_DATA_DIR, max_workers=16)
        for det in feats:
            tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], overrides, False) for p in planar]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                res = list(ex.map(_h, tasks))
            saved[("H", det)] = [(t[0]["scene"], g, r) for t, (g, r) in zip(tasks, res)]
            report(f"[H] detector matches {det} (PROSAC)", res, names, [2.5, 5.0, 10.0], 1)

    if "PnP" in args.problems:
        probs = load_7scenes_pnp_problems(DEFAULT_DATA_DIR, detectors=tuple(feats), split=args.split)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_pnp, [(p, overrides) for p in probs]))
        saved[("PnP", "all")] = [(p["scene"], g, r) for p, (g, r) in zip(probs, res)]
        report("[PnP] 7Scenes dev", res, names, [2.0, 5.0, 10.0], 1)

    if "Rigid" in args.problems:
        probs = load_7scenes_rigid_problems(DEFAULT_DATA_DIR, detectors=tuple(feats), split=args.split)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_rigid, [(p, overrides) for p in probs]))
        saved[("Rigid", "all")] = [(p["scene"], g, r) for p, (g, r) in zip(probs, res)]
        report("[Rigid] 7Scenes dev (baseline column = Procrustes RANSAC)", res, names, [2.0, 5.0, 10.0], 1)

    if args.save:
        import pickle
        with open(args.save, "wb") as f:
            pickle.dump({"configs": overrides, "results": saved}, f)
        print(f"saved per-pair results to {args.save}")


if __name__ == "__main__":
    main()

