"""Failure taxonomy on the dev split: objective failure vs. search failure.

For every pair where SupeRANSAC (defaults) fails (pose error >= 10 deg) while PoseLib or the oracle succeeds,
compare the robust scores of the models (computed in pixels with the same 1.5 px threshold):
  * count  : number of correspondences with Sampson error < threshold
  * MSAC   : sum over inliers of (1 - r^2 / threshold^2)
If the correct model (PoseLib's, or the oracle = GT F refined on its GT-consistent points) scores HIGHER than
SupeRANSAC's returned model, the objective was fine and the search never reached the correct basin
("search failure"); if it scores LOWER, the objective itself prefers the wrong model ("objective failure").

Usage: python tests/dev_taxonomy.py F|E LoMa|splg|sift
"""
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2, poselib, pysuperansac
import test_relative_pose as trp
from datasets.benchmark_pairs import DEFAULT_DATA_DIR, canonical_detector_name, ensure_cached_matches, load_calibrated_pairs, sampson_errors

TH = 1.5


def trp_skew(t):
    t = np.asarray(t, dtype=np.float64).flatten()
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float64)


def scores(corrs, F):
    if F is None or not np.all(np.isfinite(F)):
        return 0, 0.0
    r = sampson_errors(corrs, F / (np.linalg.norm(F) + 1e-15))
    m = r < TH
    return int(m.sum()), float(np.sum(1.0 - (r[m] / TH) ** 2))


def work(args):
    problem, p, corrs, sc, cfg_over = args
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    K1, K2 = np.ascontiguousarray(p["K1"]), np.ascontiguousarray(p["K2"])
    bbox = np.array([p["h1"], p["w1"], p["h2"], p["w2"]], dtype=np.float64)
    cfg = pysuperansac.RANSACSettings()
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    p1, p2 = np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4])
    Kinv = lambda K: np.linalg.inv(K)
    if problem == "F":
        F_su, inl, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, np.ascontiguousarray(sc), config=cfg)
        e_su = trp._pose_error_from_matrix(p, corrs, F_su, inl, True)
        F_pl, info = poselib.estimate_fundamental(p1, p2, trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
        e_pl = trp._pose_error_from_matrix(p, corrs, F_pl, np.asarray(info["inliers"], bool), True)
    else:
        E_su, inl, _, _ = pysuperansac.estimateEssentialMatrix(corrs, K1, K2, bbox, np.ascontiguousarray(sc), config=cfg)
        e_su = trp._pose_error_from_matrix(p, corrs, E_su, inl, False)
        F_su = Kinv(K2).T @ E_su @ Kinv(K1)
        cam = lambda K, w, h: {"model": "PINHOLE", "width": int(w), "height": int(h), "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
        pose, info = poselib.estimate_relative_pose(p1, p2, cam(K1, p["w1"], p["h1"]), cam(K2, p["w2"], p["h2"]), trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
        er, et = trp.evaluate_R_t(p["R"], p["t"], pose.R, pose.t)
        e_pl = float(max(er, et))
        E_pl = trp_skew(pose.t) @ pose.R
        F_pl = Kinv(K2).T @ E_pl @ Kinv(K1)
    # oracle: GT F, and GT F refined (Cauchy LM) on its GT-consistent correspondences
    F_gt = p["F_gt"]
    gt_mask = sampson_errors(corrs, F_gt) < TH
    F_or = F_gt.copy()
    if gt_mask.sum() >= 8:
        F_or, _ = poselib.refine_fundamental(p1[gt_mask], p2[gt_mask], F_gt, {"loss_type": "CAUCHY", "loss_scale": 0.75, "max_iterations": 100})
    return {
        "dataset": p["dataset"], "scene": p["scene"], "N": len(corrs), "gt_cons": int(gt_mask.sum()),
        "e_su": e_su, "e_pl": e_pl,
        "su": scores(corrs, F_su), "pl": scores(corrs, F_pl), "gt": scores(corrs, F_gt), "or": scores(corrs, F_or),
    }


if __name__ == "__main__":
    problem, det = sys.argv[1], canonical_detector_name(sys.argv[2])
    cfg_over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    pairs = load_calibrated_pairs(DEFAULT_DATA_DIR, split="dev", verbose=False)
    cache = ensure_cached_matches(pairs, [det], data_dir=DEFAULT_DATA_DIR)
    tasks = [(problem, p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], cfg_over) for p in pairs if len(cache[p["cache_key"]][det][0]) >= 8]
    with ProcessPoolExecutor(max_workers=24) as ex:
        R = list(ex.map(work, tasks))
    fail = 10.0
    su_fail = [r for r in R if r["e_su"] >= fail]
    print(f"[{problem} {det}] pairs={len(R)}  SU fails={len(su_fail)}  PL fails={sum(r['e_pl'] >= fail for r in R)}  both fail={sum(r['e_su'] >= fail and r['e_pl'] >= fail for r in R)}")
    cats = {"search(PL better score)": 0, "objective(PL worse score)": 0, "PL also fails": 0}
    cats_or = {"search(oracle better)": 0, "objective(oracle worse)": 0, "too few GT inliers(<15)": 0}
    for r in su_fail:
        if r["e_pl"] < fail:
            cats["search(PL better score)" if r["pl"][1] > r["su"][1] else "objective(PL worse score)"] += 1
        else:
            cats["PL also fails"] += 1
        if r["gt_cons"] < 15:
            cats_or["too few GT inliers(<15)"] += 1
        else:
            cats_or["search(oracle better)" if r["or"][1] > r["su"][1] else "objective(oracle worse)"] += 1
    print("  vs PoseLib (MSAC):", cats)
    print("  vs oracle  (MSAC):", cats_or)
    # the same for PoseLib failures that SupeRANSAC solves
    pl_fail = [r for r in R if r["e_pl"] >= fail and r["e_su"] < fail]
    print(f"  PoseLib-only failures: {len(pl_fail)} (SU better MSAC in {sum(r['su'][1] > r['pl'][1] for r in pl_fail)})")
    print("  examples (SU fail):")
    for r in sorted(su_fail, key=lambda r: -r["gt_cons"])[:12]:
        print(f"    {r['dataset'][:12]:12s} {r['scene'][:34]:34s} N={r['N']:4d} gt={r['gt_cons']:4d} e_su={r['e_su']:6.1f} e_pl={r['e_pl']:6.1f}  "
              f"cnt/msac su={r['su'][0]}/{r['su'][1]:.1f} pl={r['pl'][0]}/{r['pl'][1]:.1f} or={r['or'][0]}/{r['or'][1]:.1f}")
