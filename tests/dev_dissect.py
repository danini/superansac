"""Dissect specific failing dev pairs under several SupeRANSAC configurations."""
import os, sys, json
import numpy as np

TESTS = "/usr/local/google/home/dbarath/ransac/superansac/tests"
sys.path.insert(0, TESTS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import poselib, pysuperansac
import dev_ablate as da
import test_relative_pose as trp
from datasets.benchmark_pairs import DEFAULT_DATA_DIR, canonical_detector_name, ensure_cached_matches, load_calibrated_pairs, sampson_errors

problem = sys.argv[1]
det = canonical_detector_name(sys.argv[2])
scenes = sys.argv[3].split(",")
configs = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {"fb2": {"final_bundle_rounds": 2}}
pairs = [p for p in load_calibrated_pairs(DEFAULT_DATA_DIR, split="dev", verbose=False) if any(s in p["scene"] for s in scenes)]
cache = ensure_cached_matches(pairs, [det], data_dir=DEFAULT_DATA_DIR)

for p in pairs:
    corrs, scores = cache[p["cache_key"]][det]
    corrs = np.ascontiguousarray(corrs, dtype=np.float64)
    # GT inlier count at 1.5px Sampson wrt GT F
    gt_in = int(np.sum(sampson_errors(corrs, p["F_gt"]) < 1.5))
    print(f"\n### {p['scene']}  N={len(corrs)}  GT-consistent(<1.5px)={gt_in}  score med={np.median(scores):.2f}")
    bbox = np.array([p["h1"], p["w1"], p["h2"], p["w2"]], dtype=np.float64)
    K1, K2 = np.ascontiguousarray(p["K1"]), np.ascontiguousarray(p["K2"])
    for name, ov in configs.items():
        errs, ninl = [], []
        for seed in range(3):
            cfg = da.make_cfg(ov)
            if problem == "F":
                M, inl, _, it = pysuperansac.estimateFundamentalMatrix(corrs, bbox, np.ascontiguousarray(scores), config=cfg)
                e = trp._pose_error_from_matrix(p, corrs, M, inl, True)
            else:
                M, inl, _, it = pysuperansac.estimateEssentialMatrix(corrs, K1, K2, bbox, np.ascontiguousarray(scores), config=cfg)
                e = trp._pose_error_from_matrix(p, corrs, M, inl, False)
            errs.append(e); ninl.append(len(inl))
        print(f"  {name:22s} err={np.round(errs,2)}  inliers={ninl}")
    p1, p2 = np.ascontiguousarray(corrs[:, :2]), np.ascontiguousarray(corrs[:, 2:4])
    if problem == "F":
        M, info = poselib.estimate_fundamental(p1, p2, trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
        e = trp._pose_error_from_matrix(p, corrs, M, np.asarray(info["inliers"], bool), True)
    else:
        cam = lambda K, w, h: {"model": "PINHOLE", "width": int(w), "height": int(h), "params": [float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])]}
        pose, info = poselib.estimate_relative_pose(p1, p2, cam(K1, p["w1"], p["h1"]), cam(K2, p["w2"], p["h2"]), trp._poselib_opts(5000, 1000), trp.POSELIB_BUNDLE)
        er, et = trp.evaluate_R_t(p["R"], p["t"], pose.R, pose.t); e = max(er, et)
    print(f"  {'PoseLib':22s} err={e:.2f}  inliers={int(np.sum(info['inliers']))}  iters={info.get('iterations')}")

