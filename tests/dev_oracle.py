"""F/E headroom diagnostics on the dev split (no test data).

For every dev pair and matcher it compares, by AUC@10 (pose from cv2.recoverPose on the model's inliers, as in the
benchmark):
  SU        SupeRANSAC with default settings
  PL        PoseLib (benchmark configuration)
  GT-basin  the final robust refinement started from the ground-truth model on all matches: 3 rounds of
            re-collection (2x -> 1x threshold) + PoseLib Cauchy LM (scale 0.5 x threshold) -- the optimum of the
            current objective closest to the ground truth
  GT-LSQ    least squares on the ground-truth-consistent matches (Sampson < threshold w.r.t. the GT model)
On the pairs where SupeRANSAC fails (>= 10 deg) it also reports how often the GT-basin model scores higher than
SupeRANSAC's model under: inlier count, MSAC, and confidence-weighted MSAC (sum of c_i (1 - r_i^2 / th^2)).

Usage: python tests/dev_oracle.py F|E [matchers ...]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
import cv2
import numpy as np
import poselib
import pysuperansac

import test_relative_pose as trp
from benchmark_protocol import TABLE_MAX_ITERS, TABLE_MIN_ITERS, default_workers
from datasets.benchmark_pairs import DEFAULT_DATA_DIR, canonical_detector_name, ensure_cached_matches, load_calibrated_pairs, sampson_errors
from evaluation import pose_auc

TH = trp.THRESHOLD_PX
SCHEDULE = [2.0, 2.0 ** 0.5, 1.0]


def skew(t):
    t = np.asarray(t).flatten()
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def scores(corrs, conf, F):
    r = sampson_errors(corrs, F)
    m = r < TH
    msac = 1.0 - (r[m] / TH) ** 2
    return int(m.sum()), float(msac.sum()), float((conf[m] * msac).sum())


def refine_F(corrs, F, loss):
    x1, x2 = np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4])
    F2, _ = poselib.refine_fundamental(x1, x2, F, loss)
    return F2


def refine_E(corrs, K1, K2, E, loss):
    """Pose-space refinement in normalised coordinates (identity cameras), returns E = [t]x R."""
    n1 = cv2.undistortPoints(corrs[:, :2].reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    n2 = cv2.undistortPoints(corrs[:, 2:4].reshape(-1, 1, 2), K2, None).reshape(-1, 2)
    _, R, t, _ = cv2.recoverPose(E, n1.reshape(-1, 1, 2), n2.reshape(-1, 1, 2))
    pose = poselib.CameraPose()
    pose.R, pose.t = R, t.flatten()
    cam = {"model": "PINHOLE", "width": 2, "height": 2, "params": [1.0, 1.0, 0.0, 0.0]}
    f = 0.25 * (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1])
    opt = dict(loss)
    if "loss_scale" in opt:
        opt["loss_scale"] = opt["loss_scale"] / f
    pose2, _ = poselib.refine_relative_pose(np.ascontiguousarray(n1), np.ascontiguousarray(n2), pose, cam, cam, opt)
    return skew(pose2.t) @ pose2.R


def evaluate(args):
    pair, corrs, conf, prob = args
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    K1, K2 = pair["K1"], pair["K2"]
    E_gt = skew(pair["t"]) @ pair["R"]
    F_gt = np.linalg.inv(K2).T @ E_gt @ np.linalg.inv(K1)
    to_F = (lambda M: M) if prob == "F" else (lambda E: np.linalg.inv(K2).T @ E @ np.linalg.inv(K1))
    out = {"scene": pair["scene"]}
    bbox = np.array([pair["h1"], pair["w1"], pair["h2"], pair["w2"]], dtype=np.float64)
    cfg = pysuperansac.RANSACSettings()
    cfg.min_iterations, cfg.max_iterations = TABLE_MIN_ITERS, TABLE_MAX_ITERS
    if prob == "F":
        M, inl, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, np.ascontiguousarray(conf), config=cfg)
    else:
        M, inl, _, _ = pysuperansac.estimateEssentialMatrix(corrs, np.ascontiguousarray(K1), np.ascontiguousarray(K2), bbox,
                                                            np.ascontiguousarray(conf), config=cfg)
    M = np.asarray(M)
    out["SU"] = trp._pose_error_from_matrix(pair, corrs, M, inl, prob == "F")
    pl = (trp._eval_f_pair if prob == "F" else trp._eval_e_pair)((pair, corrs, conf, TABLE_MAX_ITERS, TABLE_MIN_ITERS, False))
    out["PL"] = pl["PoseLib"][0]
    # GT basin: robust re-collection + Cauchy LM from the ground truth
    model = F_gt.copy() if prob == "F" else E_gt.copy()
    cauchy = {"loss_type": "CAUCHY", "loss_scale": 0.5 * TH, "max_iterations": 100}
    for mult in SCHEDULE:
        sel = sampson_errors(corrs, to_F(model)) < TH * mult
        if sel.sum() < 8:
            break
        model = refine_F(corrs[sel], model, cauchy) if prob == "F" else refine_E(corrs[sel], K1, K2, model, cauchy)
    sel = np.flatnonzero(sampson_errors(corrs, to_F(model)) < TH)
    out["GT-basin"] = trp._pose_error_from_matrix(pair, corrs, model, sel, prob == "F") if len(sel) >= 8 else 180.0
    gt_sel = sampson_errors(corrs, F_gt) < TH
    out["n_gt"] = int(gt_sel.sum())
    if gt_sel.sum() >= 8:
        lsq = {"loss_type": "TRIVIAL", "max_iterations": 100}
        m2 = refine_F(corrs[gt_sel], F_gt, lsq) if prob == "F" else refine_E(corrs[gt_sel], K1, K2, E_gt, lsq)
        out["GT-LSQ"] = trp._pose_error_from_matrix(pair, corrs, m2, np.flatnonzero(gt_sel), prob == "F")
    else:
        out["GT-LSQ"] = 180.0
    out["scores_su"] = scores(corrs, conf, to_F(M))
    out["scores_gt"] = scores(corrs, conf, to_F(model))
    return out


def main():
    prob = sys.argv[1]
    dets = [canonical_detector_name(d) for d in (sys.argv[2:] or ["LoMa", "RoMa", "LoFTR", "splg", "sift"])]
    pairs = load_calibrated_pairs(DEFAULT_DATA_DIR, split="dev", verbose=False)
    cache = ensure_cached_matches(pairs, dets, data_dir=DEFAULT_DATA_DIR, max_workers=1)
    auc = lambda e: 100.0 * pose_auc(np.asarray(e), [10.0])[0]
    for det in dets:
        tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], prob) for p in pairs if len(cache[p["cache_key"]][det][0]) >= 8]
        with ProcessPoolExecutor(default_workers()) as ex:
            res = list(ex.map(evaluate, tasks))
        e = {k: np.array([r[k] for r in res]) for k in ("SU", "PL", "GT-basin", "GT-LSQ")}
        solved = (e["SU"] < 10) & (e["GT-basin"] < 10)
        print(f"[{prob} {det}] n={len(res)}  AUC@10: SU {auc(e['SU']):5.2f} | PL {auc(e['PL']):5.2f} | GT-basin {auc(e['GT-basin']):5.2f} | "
              f"GT-LSQ {auc(e['GT-LSQ']):5.2f}   median err on {solved.sum()} jointly solved: SU {np.median(e['SU'][solved]):.2f} "
              f"GT-basin {np.median(e['GT-basin'][solved]):.2f} GT-LSQ {np.median(e['GT-LSQ'][solved]):.2f}")
        fail = [r for r in res if r["SU"] >= 10 and r["GT-basin"] < 10 and r["n_gt"] >= 15]
        if fail:
            wins = [np.mean([r["scores_gt"][i] > r["scores_su"][i] for r in fail]) for i in range(3)]
            print(f"    SU failures where the GT basin is solvable: {len(fail)}; GT-basin model scores higher by "
                  f"count {100 * wins[0]:.0f}% | MSAC {100 * wins[1]:.0f}% | confidence-weighted MSAC {100 * wins[2]:.0f}%")


if __name__ == "__main__":
    main()

