"""Multicore (40-worker) benchmark comparing SupeRANSAC vs. PoseLib (Section 1 of README)."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import poselib

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


def _eval_one_fundamental_poselib(args):
    item, max_epipolar_error, progressive = args
    corrs = item["corrs"]
    K1, K2 = item["K1"], item["K2"]
    R_gt, t_gt = item["R_gt"], item["t_gt"]
    val_pts = item["val_pts"]

    x1 = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
    x2 = np.ascontiguousarray(corrs[:, 2:4], dtype=np.float64)
    ransac_opt = {
        "max_iterations": 5000,
        "min_iterations": 1000,
        "success_prob": 0.9999,
        "max_epipolar_error": max_epipolar_error,
        "progressive_sampling": progressive,
    }
    bundle_opt = {
        "loss_type": "CAUCHY",
        "loss_scale": 0.5 * max_epipolar_error,
        "max_iterations": 100,
    }

    t0 = time.perf_counter()
    F_est, info = poselib.estimate_fundamental(x1, x2, ransac_opt, bundle_opt)
    dt = (time.perf_counter() - t0) * 1000.0

    inliers = np.asarray(info.get("inliers", []), dtype=bool)
    if F_est is None or np.all(F_est == 0) or np.any(np.isnan(F_est)) or np.sum(inliers) < 7:
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


def eval_fundamental_poselib(pairs, pool, max_epipolar_error=1.5, progressive=False):
    tasks = [(p, max_epipolar_error, progressive) for p in pairs]
    out = list(pool.map(_eval_one_fundamental_poselib, tasks))
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


def _eval_one_essential_poselib(args):
    item, max_epipolar_error, progressive = args
    corrs = item["corrs"]
    img_sizes = item["image_sizes"]
    K1, K2 = item["K1"], item["K2"]
    R_gt, t_gt = item["R_gt"], item["t_gt"]

    x1 = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
    x2 = np.ascontiguousarray(corrs[:, 2:4], dtype=np.float64)
    cam1 = {
        "model": "PINHOLE",
        "width": int(max(img_sizes[0], K1[0, 2] * 2)),
        "height": int(max(img_sizes[1], K1[1, 2] * 2)),
        "params": [float(K1[0, 0]), float(K1[1, 1]), float(K1[0, 2]), float(K1[1, 2])],
    }
    cam2 = {
        "model": "PINHOLE",
        "width": int(max(img_sizes[2], K2[0, 2] * 2)),
        "height": int(max(img_sizes[3], K2[1, 2] * 2)),
        "params": [float(K2[0, 0]), float(K2[1, 1]), float(K2[0, 2]), float(K2[1, 2])],
    }
    ransac_opt = {
        "max_iterations": 5000,
        "min_iterations": 1000,
        "success_prob": 0.9999,
        "max_epipolar_error": max_epipolar_error,
        "progressive_sampling": progressive,
    }
    bundle_opt = {
        "loss_type": "CAUCHY",
        "loss_scale": 0.5 * max_epipolar_error,
        "max_iterations": 100,
    }

    t0 = time.perf_counter()
    pose, info = poselib.estimate_relative_pose(x1, x2, cam1, cam2, ransac_opt, bundle_opt)
    dt = (time.perf_counter() - t0) * 1000.0

    inliers = np.asarray(info.get("inliers", []), dtype=bool)
    if pose is None or np.sum(inliers) < 5:
        return 180.0, 180.0, 180.0, dt

    R_est = pose.R
    t_est = pose.t
    eR = rotation_error_deg(R_gt, R_est)
    et = angular_translation_error_deg(t_gt, t_est)
    return max(eR, et), eR, et, dt


def eval_essential_poselib(pairs, pool, max_epipolar_error=1.5, progressive=False):
    tasks = [(p, max_epipolar_error, progressive) for p in pairs]
    out = list(pool.map(_eval_one_essential_poselib, tasks))
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


def _eval_one_homography_poselib(args):
    item, max_reproj_error, progressive = args
    corrs = item["corrs"]
    H_gt = item["H_gt"]
    corners = item["corners"]
    val_pts = item["val_pts"]

    x1 = np.ascontiguousarray(corrs[:, :2], dtype=np.float64)
    x2 = np.ascontiguousarray(corrs[:, 2:4], dtype=np.float64)
    ransac_opt = {
        "max_iterations": 5000,
        "min_iterations": 1000,
        "success_prob": 0.9999,
        "max_reproj_error": max_reproj_error,
        "progressive_sampling": progressive,
    }
    bundle_opt = {
        "loss_type": "CAUCHY",
        "loss_scale": 0.5 * max_reproj_error,
        "max_iterations": 100,
    }

    t0 = time.perf_counter()
    H_est, info = poselib.estimate_homography(x1, x2, ransac_opt, bundle_opt)
    dt = (time.perf_counter() - t0) * 1000.0

    inliers = np.asarray(info.get("inliers", []), dtype=bool)
    if H_est is None or np.all(H_est == 0) or np.any(np.isnan(H_est)) or np.sum(inliers) < 4:
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


def eval_homography_poselib(pairs, pool, max_reproj_error=4.0, progressive=False):
    tasks = [(p, max_reproj_error, progressive) for p in pairs]
    out = list(pool.map(_eval_one_homography_poselib, tasks))
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


def main():
    with open(BENCHMARK_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    with open(FINAL_RESULTS_PATH, "r") as f:
        super_res = json.load(f)

    workers = min(40, os.cpu_count() or 40)
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    poselib_variants = {}

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        # Evaluate PoseLib under both default and matched thresholds, Uniform and PROSAC
        poselib_variants["fundamental_1.0px_unif"] = eval_fundamental_poselib(data["fundamental"], pool, 1.0, False)
        poselib_variants["fundamental_1.5px_unif"] = eval_fundamental_poselib(data["fundamental"], pool, 1.5, False)
        poselib_variants["fundamental_1.5px_prosac"] = eval_fundamental_poselib(data["fundamental"], pool, 1.5, True)

        poselib_variants["essential_1.0px_unif"] = eval_essential_poselib(data["essential"], pool, 1.0, False)
        poselib_variants["essential_1.5px_unif"] = eval_essential_poselib(data["essential"], pool, 1.5, False)
        poselib_variants["essential_1.5px_prosac"] = eval_essential_poselib(data["essential"], pool, 1.5, True)

        poselib_variants["homography_4.0px_unif"] = eval_homography_poselib(data["homography"], pool, 4.0, False)
        poselib_variants["homography_4.0px_prosac"] = eval_homography_poselib(data["homography"], pool, 4.0, True)
        poselib_variants["homography_3.0px_unif"] = eval_homography_poselib(data["homography"], pool, 3.0, False)

    t_wall = time.perf_counter() - t0
    print(f"Evaluated all PoseLib variants on {workers} workers in {t_wall:.2f}s\n")

    print("=== PoseLib Variant Breakdown ===")
    for k, v in poselib_variants.items():
        print(f"  {k:28s}: {v['primary_metric']:16s} = {v['primary_auc']:.4f}  (mean {v['mean_time_ms']:6.2f} ms, median {v['median_time_ms']:6.2f} ms)")

    best_poselib = {
        "fundamental": max(
            [poselib_variants["fundamental_1.0px_unif"], poselib_variants["fundamental_1.5px_unif"], poselib_variants["fundamental_1.5px_prosac"]],
            key=lambda x: x["primary_auc"],
        ),
        "essential": max(
            [poselib_variants["essential_1.0px_unif"], poselib_variants["essential_1.5px_unif"], poselib_variants["essential_1.5px_prosac"]],
            key=lambda x: x["primary_auc"],
        ),
        "homography": max(
            [poselib_variants["homography_4.0px_unif"], poselib_variants["homography_4.0px_prosac"], poselib_variants["homography_3.0px_unif"]],
            key=lambda x: x["primary_auc"],
        ),
    }

    with open(POSELIB_RESULTS_PATH, "w") as f:
        json.dump({"variants": poselib_variants, "best": best_poselib}, f, indent=2)

    print("\n=== Section 1 Head-to-Head Summary: PoseLib (Best) vs. SupeRANSAC ===")
    header = f"{'Problem':14s} | {'Primary Metric':20s} | {'PoseLib (Best)':18s} | {'SupeRANSAC':18s} | {'Gain vs PoseLib':18s}"
    print(header)
    print("-" * len(header))
    for prob in ["fundamental", "essential", "homography", "rigid_pose"]:
        m_name = super_res[prob]["primary_metric"]
        s_auc = super_res[prob]["primary_auc"]
        s_str = f"{s_auc:.4f} ({super_res[prob]['mean_time_ms']:5.2f}ms)"
        if prob in best_poselib:
            p_auc = best_poselib[prob]["primary_auc"]
            p_str = f"{p_auc:.4f} ({best_poselib[prob]['mean_time_ms']:5.2f}ms)"
            diff_pt = (s_auc - p_auc) * 100.0
            rel_pct = ((s_auc - p_auc) / max(p_auc, 1e-9)) * 100.0
            vs_pl = f"{diff_pt:+.2f}pt ({rel_pct:+.1f}%)"
        else:
            p_str = "N/A (no 3D-3D)"
            vs_pl = "N/A"
        print(f"{prob:14s} | {m_name:20s} | {p_str:18s} | {s_str:18s} | {vs_pl:18s}")


if __name__ == "__main__":
    main()

