"""Cross-refinement check on one pair: does the model or the inlier set cause the gap?"""
import os, sys, json
import numpy as np
TESTS = "/usr/local/google/home/dbarath/ransac/superansac/tests"
sys.path.insert(0, TESTS)
import poselib, pysuperansac, cv2
import test_relative_pose as trp
from datasets.benchmark_pairs import DEFAULT_DATA_DIR, canonical_detector_name, ensure_cached_matches, load_calibrated_pairs, sampson_errors

problem, det, scene = sys.argv[1], canonical_detector_name(sys.argv[2]), sys.argv[3]
p = [q for q in load_calibrated_pairs(DEFAULT_DATA_DIR, split="dev", verbose=False) if scene in q["scene"]][0]
cache = ensure_cached_matches([p], [det], data_dir=DEFAULT_DATA_DIR)
corrs, scores = cache[p["cache_key"]][det]
corrs = np.ascontiguousarray(corrs, dtype=np.float64)
bbox = np.array([p["h1"], p["w1"], p["h2"], p["w2"]], dtype=np.float64)
K1, K2 = np.ascontiguousarray(p["K1"]), np.ascontiguousarray(p["K2"])
p1, p2 = np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4])
gt_mask = sampson_errors(corrs, p["F_gt"]) < 1.5
print(f"{p['scene']} N={len(corrs)} GT-consistent={gt_mask.sum()}")

def err_F(F, mask):
    return trp._pose_error_from_matrix(p, corrs, F, np.asarray(mask, bool), True)

cfg = pysuperansac.RANSACSettings(); cfg.min_iterations, cfg.max_iterations = 1000, 5000; cfg.final_bundle_rounds = 2
F_su, inl_su, _, _ = pysuperansac.estimateFundamentalMatrix(corrs, bbox, np.ascontiguousarray(scores), config=cfg)
m_su = np.zeros(len(corrs), bool); m_su[np.asarray(inl_su, int)] = True
F_pl, info = poselib.estimate_fundamental(p1, p2, trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
m_pl = np.asarray(info["inliers"], bool)
print(f"SU: err={err_F(F_su, m_su):.2f} inl={m_su.sum()} (GT-inl among them {np.sum(m_su & gt_mask)}) | PL: err={err_F(F_pl, m_pl):.2f} inl={m_pl.sum()} (GT {np.sum(m_pl & gt_mask)})")
print(f"inlier set overlap: both={np.sum(m_su & m_pl)} SU-only={np.sum(m_su & ~m_pl)} PL-only={np.sum(m_pl & ~m_su)}")
# residual stats of each F on the GT-consistent points
for name, F in (("SU", F_su), ("PL", F_pl)):
    r = sampson_errors(corrs[gt_mask], F)
    print(f"  {name} F on GT-consistent pts: mean {r.mean():.3f} med {np.median(r):.3f} max {r.max():.2f}; on GT-F residual of SU/PL inliers")
# cross refinement with PoseLib python refine
bo = {"loss_type": "CAUCHY", "loss_scale": 0.75, "max_iterations": 100}
for name, F0, mask in (("SU model, SU inliers", F_su, m_su), ("SU model, PL inliers", F_su, m_pl), ("PL model, SU inliers", F_pl, m_su), ("PL model, PL inliers", F_pl, m_pl), ("SU model, GT inliers", F_su, gt_mask), ("PL model, GT inliers", F_pl, gt_mask)):
    Fr, st = poselib.refine_fundamental(p1[mask], p2[mask], F0, bo)
    print(f"  refine({name}): err {err_F(F0, mask):.2f} -> {err_F(Fr, mask):.2f}")
# error of the GT F itself through the same metric (sanity: how well-posed is pose-from-F here?)
print(f"GT F through metric with GT inliers: {err_F(p['F_gt'], gt_mask):.2f}")
# 8-point LSQ on GT inliers
F8, _ = cv2.findFundamentalMat(p1[gt_mask], p2[gt_mask], cv2.FM_8POINT)
print(f"8-point on GT inliers: {err_F(F8, gt_mask):.2f}")

