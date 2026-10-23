"""Unified multicore benchmark runner for SupeRANSAC across Fundamental, Essential, Homography, and Rigid Pose."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT_DIR = "/usr/local/google/home/dbarath/ransac"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DATA_PATH = (
    os.path.join(SCRIPT_DIR, "benchmark_data.pkl")
    if os.path.exists(os.path.join(SCRIPT_DIR, "benchmark_data.pkl"))
    else os.path.join(ROOT_DIR, "benchmark", "benchmark_data.pkl")
)
FINAL_RESULTS_PATH = os.path.join(SCRIPT_DIR, "final_results.json")
POSELIB_RESULTS_PATH = os.path.join(SCRIPT_DIR, "poselib_results.json")

_PYSUPERANSAC = None


def compute_auc(errors, thresholds):
    errors = np.asarray(errors, dtype=np.float64)
    errors = np.where(np.isnan(errors) | np.isinf(errors), 1e9, errors)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs


def rotation_error_deg(R_gt, R_est):
    R_rel = np.matmul(R_gt.T, R_est)
    tr = np.trace(R_rel)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def angular_translation_error_deg(t_gt, t_est):
    t1 = t_gt.flatten()
    t2 = t_est.flatten()
    n1 = np.linalg.norm(t1)
    n2 = np.linalg.norm(t2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0
    c = np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def sampson_errors(pts, F):
    x1 = np.hstack([pts[:, :2], np.ones((len(pts), 1))])
    x2 = np.hstack([pts[:, 2:4], np.ones((len(pts), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-15
    return np.abs(x2Fx1) / np.sqrt(denom)


def _eval_one_fundamental(item):
    corrs = item["corrs"]
    img_sizes = np.array(item["image_sizes"], dtype=np.float64)
    probs = item["scores"]
    K1, K2 = item["K1"], item["K2"]
    R_gt, t_gt = item["R_gt"], item["t_gt"]
    val_pts = item["val_pts"]

    cfg = _PYSUPERANSAC.RANSACSettings()
    t0 = time.perf_counter()
    F_est, inliers, score, iters = _PYSUPERANSAC.estimateFundamentalMatrix(
        corrs, img_sizes, probs, config=cfg
    )
    dt = (time.perf_counter() - t0) * 1000.0

    if F_est is None or np.all(F_est == 0) or np.any(np.isnan(F_est)) or len(inliers) < 7:
        return 180.0, 1e6, dt

    s_err = float(np.mean(sampson_errors(val_pts, F_est)))
    E_est = K2.T @ F_est @ K1
    inl_pts = corrs[inliers]
    p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
    p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
    _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
    eR = rotation_error_deg(R_gt, R_est)
    et = angular_translation_error_deg(t_gt, t_est)
    return max(eR, et), s_err, dt


def eval_fundamental(pairs, pool):
    out = list(pool.map(_eval_one_fundamental, pairs))
    pose_errors = [x[0] for x in out]
    sampson_rmse = [x[1] for x in out]
    runtimes_ms = [x[2] for x in out]

    auc5, auc10, auc20 = compute_auc(pose_errors, [5.0, 10.0, 20.0])
    s_auc1, s_auc25, s_auc5 = compute_auc(sampson_rmse, [1.0, 2.5, 5.0])
    return {
        "primary_metric": "Pose AUC@10deg",
        "primary_auc": auc10,
        "pose_auc_5": auc5,
        "pose_auc_10": auc10,
        "pose_auc_20": auc20,
        "sampson_auc_1px": s_auc1,
        "sampson_auc_2_5px": s_auc25,
        "sampson_auc_5px": s_auc5,
        "mean_time_ms": float(np.mean(runtimes_ms)),
        "median_time_ms": float(np.median(runtimes_ms)),
    }


def _eval_one_essential(item):
    corrs = item["corrs"]
    img_sizes = np.array(item["image_sizes"], dtype=np.float64)
    probs = item["scores"]
    K1, K2 = item["K1"], item["K2"]
    R_gt, t_gt = item["R_gt"], item["t_gt"]

    cfg = _PYSUPERANSAC.RANSACSettings()
    t0 = time.perf_counter()
    E_est, inliers, score, iters = _PYSUPERANSAC.estimateEssentialMatrix(
        corrs, K1, K2, img_sizes, probs, config=cfg
    )
    dt = (time.perf_counter() - t0) * 1000.0

    if E_est is None or np.all(E_est == 0) or np.any(np.isnan(E_est)) or len(inliers) < 5:
        return 180.0, 180.0, 180.0, dt

    inl_pts = corrs[inliers]
    p1_n = cv2.undistortPoints(inl_pts[:, :2].reshape(-1, 1, 2), K1, None)
    p2_n = cv2.undistortPoints(inl_pts[:, 2:4].reshape(-1, 1, 2), K2, None)
    _, R_est, t_est, _ = cv2.recoverPose(E_est, p1_n, p2_n)
    eR = rotation_error_deg(R_gt, R_est)
    et = angular_translation_error_deg(t_gt, t_est)
    return max(eR, et), eR, et, dt


def eval_essential(pairs, pool):
    out = list(pool.map(_eval_one_essential, pairs))
    pose_errors = [x[0] for x in out]
    rot_errors = [x[1] for x in out]
    trans_errors = [x[2] for x in out]
    runtimes_ms = [x[3] for x in out]

    auc5, auc10, auc20 = compute_auc(pose_errors, [5.0, 10.0, 20.0])
    r_auc10 = compute_auc(rot_errors, [10.0])[0]
    t_auc10 = compute_auc(trans_errors, [10.0])[0]
    return {
        "primary_metric": "Pose AUC@10deg",
        "primary_auc": auc10,
        "pose_auc_5": auc5,
        "pose_auc_10": auc10,
        "pose_auc_20": auc20,
        "rot_auc_10": r_auc10,
        "trans_auc_10": t_auc10,
        "mean_time_ms": float(np.mean(runtimes_ms)),
        "median_time_ms": float(np.median(runtimes_ms)),
    }


def _eval_one_homography(item):
    corrs = item["corrs"]
    img_sizes = np.array(item["image_sizes"], dtype=np.float64)
    probs = item["scores"]
    H_gt = item["H_gt"]
    corners = item["corners"]
    val_pts = item["val_pts"]

    cfg = _PYSUPERANSAC.RANSACSettings()
    t0 = time.perf_counter()
    H_est, inliers, score, iters = _PYSUPERANSAC.estimateHomography(
        corrs, img_sizes, probs, config=cfg
    )
    dt = (time.perf_counter() - t0) * 1000.0

    if H_est is None or np.all(H_est == 0) or np.any(np.isnan(H_est)) or len(inliers) < 4:
        return 1e6, 1e6, dt

    c_gt = corners @ H_gt.T
    c_gt = c_gt[:, :2] / (c_gt[:, 2:3] + 1e-12)
    c_est = corners @ H_est.T
    if np.any(np.abs(c_est[:, 2]) < 1e-8):
        return 1e6, 1e6, dt
    c_est = c_est[:, :2] / c_est[:, 2:3]
    c_err = float(np.mean(np.linalg.norm(c_est - c_gt, axis=1)))

    p1_h = np.hstack([val_pts[:, :2], np.ones((len(val_pts), 1))])
    p2_est = p1_h @ H_est.T
    p2_est = p2_est[:, :2] / (p2_est[:, 2:3] + 1e-12)
    v_err = float(np.mean(np.linalg.norm(p2_est - val_pts[:, 2:4], axis=1)))
    return c_err, v_err, dt


def eval_homography(pairs, pool):
    out = list(pool.map(_eval_one_homography, pairs))
    corner_errors = [x[0] for x in out]
    val_errors = [x[1] for x in out]
    runtimes_ms = [x[2] for x in out]

    c_auc5, c_auc10, c_auc20 = compute_auc(corner_errors, [5.0, 10.0, 20.0])
    v_auc5, v_auc10, v_auc20 = compute_auc(val_errors, [5.0, 10.0, 20.0])
    return {
        "primary_metric": "Corner AUC@5px",
        "primary_auc": c_auc5,
        "corner_auc_5px": c_auc5,
        "corner_auc_10px": c_auc10,
        "corner_auc_20px": c_auc20,
        "val_auc_5px": v_auc5,
        "val_auc_10px": v_auc10,
        "val_auc_20px": v_auc20,
        "mean_time_ms": float(np.mean(runtimes_ms)),
        "median_time_ms": float(np.median(runtimes_ms)),
    }


def _eval_one_rigid_pose(item):
    corrs = item["corrs"]
    probs = item["scores"]
    T_gt = item["T_gt"]
    th = item["threshold"]

    min_coords = np.min(corrs, axis=0)
    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-min_coords[0], -min_coords[1], -min_coords[2], 1],
    ], dtype=np.float64)
    T2inv = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [min_coords[3], min_coords[4], min_coords[5], 1],
    ], dtype=np.float64)
    shifted = np.ascontiguousarray(corrs - min_coords, dtype=np.float64)
    bbox = np.ascontiguousarray(np.max(shifted, axis=0) + 1e-3, dtype=np.float64)

    cfg = _PYSUPERANSAC.RANSACSettings()
    cfg.inlier_threshold = th
    t0 = time.perf_counter()
    T_est_raw, inliers, score, iters = _PYSUPERANSAC.estimateRigidTransform(
        shifted, bbox, probs, config=cfg
    )
    dt = (time.perf_counter() - t0) * 1000.0

    if T_est_raw is None or np.all(T_est_raw == 0) or np.any(np.isnan(T_est_raw)) or len(inliers) < 3:
        return 1e6, 180.0, 1e6, dt

    T_est = (T1 @ T_est_raw @ T2inv).T
    eR = rotation_error_deg(T_gt[:3, :3], T_est[:3, :3])
    et_cm = float(np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3]) * 100.0)
    return max(eR / 5.0, et_cm / 10.0), eR, et_cm, dt


def eval_rigid_pose(pairs, pool):
    out = list(pool.map(_eval_one_rigid_pose, pairs))
    joint_norm_errors = [x[0] for x in out]
    rot_errors = [x[1] for x in out]
    trans_errors_cm = [x[2] for x in out]
    runtimes_ms = [x[3] for x in out]

    auc_joint = compute_auc(joint_norm_errors, [1.0, 2.0])[0]
    auc_r5 = compute_auc(rot_errors, [5.0])[0]
    auc_t10 = compute_auc(trans_errors_cm, [10.0])[0]
    return {
        "primary_metric": "Pose AUC@(5deg,10cm)",
        "primary_auc": auc_joint,
        "pose_auc_5deg_10cm": auc_joint,
        "rot_auc_5deg": auc_r5,
        "trans_auc_10cm": auc_t10,
        "mean_time_ms": float(np.mean(runtimes_ms)),
        "median_time_ms": float(np.median(runtimes_ms)),
    }


def main():
    global _PYSUPERANSAC
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "superansac", "build", "python"),
        help="Directory containing pysuperansac.so",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=FINAL_RESULTS_PATH,
        help="JSON output path for SupeRANSAC benchmark results (default: final_results.json)",
    )
    parser.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 40), help="Number of worker processes")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.lib_dir))
    import pysuperansac
    _PYSUPERANSAC = pysuperansac

    with open(BENCHMARK_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    results = {}
    t_wall0 = time.perf_counter()
    print(f"Evaluating pysuperansac from: {pysuperansac.__file__} (using {args.workers} workers)\n")

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        results["fundamental"] = eval_fundamental(data["fundamental"], pool)
        results["essential"] = eval_essential(data["essential"], pool)
        results["homography"] = eval_homography(data["homography"], pool)
        results["rigid_pose"] = eval_rigid_pose(data["rigid_pose"], pool)

    t_wall = time.perf_counter() - t_wall0
    print(f"Total benchmark wall-clock time: {t_wall:.2f}s\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved SupeRANSAC benchmark results to {args.output_json}\n")

    poselib_best = None
    if os.path.exists(POSELIB_RESULTS_PATH):
        with open(POSELIB_RESULTS_PATH, "r") as f:
            poselib_best = json.load(f).get("best", {})

    header = f"{'Problem':15s} | {'Primary Metric':22s} | {'PoseLib (Best)':15s} | {'SupeRANSAC':15s} | {'Gain vs PoseLib':16s} | {'Mean Time (ms)':14s}"
    print(header)
    print("-" * len(header))
    for prob in ["fundamental", "essential", "homography", "rigid_pose"]:
        r = results[prob]
        curr_auc = r["primary_auc"]
        t_ms = r["mean_time_ms"]
        if poselib_best and prob in poselib_best:
            p_auc = poselib_best[prob]["primary_auc"]
            abs_gain = (curr_auc - p_auc) * 100.0
            rel_gain = ((curr_auc - p_auc) / max(p_auc, 1e-9)) * 100.0
            print(
                f"{prob:15s} | {r['primary_metric']:22s} | {p_auc:15.4f} | {curr_auc:15.4f} | "
                f"{abs_gain:+7.2f}pt ({rel_gain:+.1f}%) | {t_ms:6.2f}"
            )
        else:
            print(
                f"{prob:15s} | {r['primary_metric']:22s} | {'N/A (3D-3D)':15s} | {curr_auc:15.4f} | "
                f"{'N/A':16s} | {t_ms:6.2f}"
            )

    print("\nDetailed Metrics:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
