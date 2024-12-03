import cv2
import h5py
import os
import numpy as np
import argparse
import yaml
import math
import time
from utils import load_h5, read_h5, append_h5
from errors import reprojection_error, homography_pose_error
from joblib import Parallel, delayed
from tqdm import tqdm
from functions import point_matching, normalize_keypoints
from evaluation import evaluate_R_t, pose_auc, qvec2rotmat
from lightglue import LightGlue, SuperPoint
from romatch import roma_outdoor
import kornia as K
import torch
from utils import detect_and_load_data

import pygcransac
import pymagsac
import poselib
import pycolmap
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

from datasets.scannet import ScanNet
from datasets.lamar import Lamar
from datasets.eth3d import ETH3D
from datasets.kitti import Kitti
from datasets.phototourism import PhotoTourism
from datasets.seven_scenes import SevenScenes

def run(matches, scores, K1, K2, R_gt, t_gt, args):
    # Initialize the errors
    rotation_error = 1e10
    translation_error = 1e10
    E_est = None
    
    # Sort the matches by matching score
    indices = np.argsort(scores)[::-1]
    matches = matches[indices, :]
    scores = scores[indices]

    # Return if there are fewer than 4 correspondences
    if matches.shape[0] < 4:
        return (np.inf, np.inf), 0, 0

    if args.method == "poselib":
        ransac_options = {"max_iterations": args.maximum_iterations,
                        "min_iterations":  args.minimum_iterations,
                        "success_prob": args.confidence,
                        "max_epipolar_error": args.inlier_threshold,
                        "progressive_sampling": args.sampler.lower() == 'prosac'}
        
        tic = time.perf_counter()
        H_est, res  = poselib.estimate_homography(matches[:,:2], matches[:,2:4], ransac_opt = ransac_options)
        inliers = np.array(res['inliers'])
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "pycolmap":
        ransac_options = pycolmap.RANSACOptions(
            max_error=args.inlier_threshold,  # for example the reprojection error in pixels
            min_inlier_ratio=0.01,
            confidence=0.9999,
            min_num_trials=args.minimum_iterations,
            max_num_trials=args.maximum_iterations,
        )
        
        xy1 = np.ascontiguousarray(matches[:, 0:2]).astype(np.float64)
        xy2 = np.ascontiguousarray(matches[:, 2:4]).astype(np.float64)
        
        tic = time.perf_counter()
        res  = pycolmap.homography_matrix_estimation(xy1, xy2, estimation_options = ransac_options)
        H_est = res['H']
        inliers = np.array(res['inliers'])
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "RANSAC OpenCV":
        # Run the homography estimation implemented in OpenCV        
        tic = time.perf_counter()
        args.confidence = 0.99999999999999
        H_est, inliers = cv2.findHomography(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.RANSAC,
            args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "RHO OpenCV":
        # Run the homography estimation implemented in OpenCV        
        tic = time.perf_counter()
        args.confidence = 0.99999999999999
        H_est, inliers = cv2.findHomography(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.RHO,
            args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "LMEDS OpenCV":
        # Run the homography estimation implemented in OpenCV        
        tic = time.perf_counter()
        args.confidence = 0.99999999999999
        H_est, inliers = cv2.findHomography(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.LMEDS,
            args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "USAC_MAGSAC OpenCV":
        # Run the homography estimation implemented in OpenCV        
        tic = time.perf_counter()
        args.confidence = 0.99999999999999
        H_est, inliers = cv2.findHomography(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.USAC_MAGSAC,
            args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "skimage":
        if matches.shape[0] < 6:
            return (np.inf, np.inf), 0, 0
        tic = time.perf_counter()
        model_robust, inliers = ransac((matches[:, :2], matches[:, 2:4]),
                                        ProjectiveTransform, min_samples=4,
                                        residual_threshold=args.inlier_threshold,
                                        max_trials=args.maximum_iterations,
                                        stop_probability=args.confidence)
        inliers = np.array(inliers).astype(bool).reshape(-1)
        H_est = model_robust.params.reshape(3,3)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "GC-RANSAC github":        
        tic = time.perf_counter()
        H_est, inliers = pygcransac.findHomography(
            np.ascontiguousarray(matches), 
            int(K1[1, 2] * 2),
            int(K1[0, 2] * 2),
            int(K2[1, 2] * 2),
            int(K2[0, 2] * 2),
            use_sprt = False,
            use_space_partitioning = False,
            probabilities = [],
            spatial_coherence_weight = 0.0,
            conf = 0.9999999999999,
            min_iters = args.minimum_iterations,
            max_iters = args.maximum_iterations,
            threshold = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "MAGSAC github":        
        tic = time.perf_counter()
        H_est, inliers = pymagsac.findHomography(
            np.ascontiguousarray(matches), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            use_magsac_plus_plus = False,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "MAGSAC++ github":        
        tic = time.perf_counter()
        H_est, inliers = pymagsac.findHomography(
            np.ascontiguousarray(matches), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            use_magsac_plus_plus = True,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "kornia":
        device = "cpu"
        batch_size = min(2500, min(args.batch_size, args.maximum_iterations))
        batched_max_iter = math.ceil(args.maximum_iterations / float(batch_size))
        # Run the homography estimation implemented in pydegensac
        tic = time.perf_counter()
        rsc = K.geometry.RANSAC('homography',  args.inlier_threshold, batch_size, batched_max_iter, 0.999999, max_lo_iters=10).to(device)
        xy1 = torch.from_numpy(matches[:, :2]).to(device=device, dtype=torch.float32)
        xy2 = torch.from_numpy(matches[:, 2:4]).to(device=device, dtype=torch.float32)
        tic = time.perf_counter()
        if args.device =='cuda':
            torch.cuda.synchronize()
        with torch.inference_mode():
            H_est, inliers = rsc(xy1, xy2)
            H_est = H_est.detach().cpu().numpy().squeeze()
            inliers = inliers.detach().cpu().numpy().reshape(-1).astype(bool)
        if args.device =='cuda':
            torch.cuda.synchronize()
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "vsac":
        import pvsac

        tic = time.perf_counter()
        params = pvsac.Params(pvsac.EstimationMethod.Homography, 
                              args.inlier_threshold, 
                              0.999999999999999999999999999999999999999, 
                              args.maximum_iterations, 
                              pvsac.SamplingMethod.SAMPLING_UNIFORM, 
                              pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        H_est, inliers = pvsac.estimate(params, matches[:,:2], matches[:,2:], None, None, None, None)
        toc = time.perf_counter()
        elapsed_time = toc - tic

        if np.sum(inliers) < 8:
            return (np.inf, np.inf), 0, 0
    elif args.method == "MAGSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        try:
            norm_matches = np.zeros(matches.shape)
            norm_matches[:, :2] = normalize_keypoints(matches[:, :2], K1)
            norm_matches[:, 2:] = normalize_keypoints(matches[:, 2:], K2)
            args.confidence = 0.99999999999999
            H_est, inliers = cv2.findHomography(
                np.ascontiguousarray(matches[:, :2]),
                np.ascontiguousarray(matches[:, 2:4]),
                cv2.USAC_MAGSAC,
                args.inlier_threshold,
                maxIters = args.maximum_iterations,
                confidence = args.confidence)
        except:
            E_est = None
            inliers = np.zeros(matches.shape[0], dtype=bool)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "GCRANSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        try:
            norm_matches = np.zeros(matches.shape)
            norm_matches[:, :2] = normalize_keypoints(matches[:, :2], K1)
            norm_matches[:, 2:] = normalize_keypoints(matches[:, 2:], K2)
            args.confidence = 0.99999999999999
            H_est, inliers = cv2.findHomography(
                np.ascontiguousarray(matches[:, :2]),
                np.ascontiguousarray(matches[:, 2:4]),
                cv2.USAC_ACCURATE,
                args.inlier_threshold,
                maxIters = args.maximum_iterations,
                confidence = args.confidence)
        except:
            E_est = None
            inliers = np.zeros(matches.shape[0], dtype=bool)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        
    if H_est is None:
        return (np.inf, np.inf), 0, elapsed_time
    
    # Count the inliers
    inlier_number = len(inliers)

    if inlier_number < 4:  
        H_est = np.identity(3)

    # Calculate the pose error of the estimated homography given the ground truth relative pose
    pose = np.zeros((3, 4)) 
    pose[:, :3] = R_gt
    pose[:, 3] = t_gt.reshape(3)
    rotation_error, translation_error, absolute_translation_error = homography_pose_error(H_est, 1.0, pose, K1, K2)

    return (rotation_error, translation_error), inlier_number, elapsed_time

if __name__ == "__main__":
    # Passing the arguments
    parser = argparse.ArgumentParser(description="Running on the HEB benchmark")
    parser.add_argument('--features', type=str, help="Choose from: DISKLG, RoMA.", choices=["splg", "RoMA"], default="RoMA")
    parser.add_argument('--batch_size', type=int, help="Batch size for multi-CPU processing", default=1000)
    parser.add_argument('--output_db_path', type=str, help="The path to where the dataset of matches should be saved.", default="/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test/matches.h5")
    parser.add_argument("--confidence", type=float, default=0.999999999999999999999999)
    parser.add_argument("--inlier_threshold", type=float, default=10.0)
    parser.add_argument("--minimum_iterations", type=int, default=1000)
    parser.add_argument("--maximum_iterations", type=int, default=1000)
    parser.add_argument("--sampler", type=str, help="Choose from: Uniform, PROSAC, PNAPSAC, Importance, ARSampler.", choices=["Uniform", "PROSAC", "PNAPSAC", "Importance", "ARSampler"], default="PNAPSAC")
    parser.add_argument("--spatial_coherence_weight", type=float, default=0.1)
    parser.add_argument("--neighborhood_size", type=float, default=20)
    parser.add_argument("--neighborhood_grid_density", type=float, default=4)
    parser.add_argument("--method", type=str, help="Choose from: poselib, pycolmap.", choices=["poselib", "pycolmap", "RANSAC OpenCV", "skimage", "LMEDS OpenCV", "pydegensac github", "kornia", "GC-RANSAC github", "MAGSAC github", "MAGSAC++ github"], default="poselib")
    parser.add_argument("--core_number", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    print(f"Testing {args.method}")

    datasets = [ScanNet(root_dir=os.path.expanduser("/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test"), split='test'),
                PhotoTourism(root_dir=os.path.expanduser("/media/hdd2tb/datasets/RANSAC-Tutorial-Data"), split='val'),
                Lamar(root_dir=os.path.expanduser("/media/hdd3tb/datasets/lamar/CAB/sessions/query_val_hololens")),
                ETH3D(root_dir=os.path.expanduser("/media/hdd2tb/datasets/eth3d"), split='test', downsize_factor=8),
                SevenScenes(root_dir=os.path.expanduser("/media/hdd2tb/datasets/7scenes"), split='test', scene='all'),
                Kitti(root_dir="/media/hdd3tb/datasets/kitti/dataset", steps=20)]

    if args.features == "splg":
        thresholds = {
            "LMEDS OpenCV": 1.0,
            "RHO OpenCV": 5.0,
            "RANSAC OpenCV": 3.0,
            "kornia": 10.0,
            "GC-RANSAC github": 3.0,
            "MAGSAC github": 10.0,
            "MAGSAC++ github": 10.0,
            "USAC_MAGSAC OpenCV": 1.0,
            "poselib": 0.5,
            "pycolmap": 5.0,
            "vsac": 2.0,
            "GCRANSAC OpenCV": 3.0,
            "MAGSAC OpenCV": 0.5
        }

        print("Initialize SP+LG detector")
        detector = SuperPoint(max_num_keypoints=2048).eval().to(args.device)  # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(args.device)  # load the matcher
    elif args.features == "RoMA":
        thresholds = {
            "LMEDS OpenCV": 1.0,
            "RHO OpenCV": 3.0,
            "RANSAC OpenCV": 3.0,
            "kornia": 0.75,
            "GC-RANSAC github": 0.75,
            "MAGSAC github": 3.0,
            "MAGSAC++ github": 5.0,
            "USAC_MAGSAC OpenCV": 0.5,
            "poselib": 0.5,
            "pycolmap": 0.75,
            "vsac": 0.5,
            "GCRANSAC OpenCV": 0.5,
            "MAGSAC OpenCV": 0.5
        }
        
        print("Initialize RoMA detector")
        detector = roma_outdoor(device = args.device)
        matcher = None
    
    for dataset in datasets:
        pose_errors = {}
        runtimes = {}
        inlier_numbers = {}
        db_name = dataset.__class__.__name__.lower()
        dataloader = dataset.get_dataloader()
        args.output_db_path = dataset.root_dir + "/matches.h5"

        processing_queue = []
        run_count = 1
        for i, data in enumerate(dataloader):
            matches, scores = detect_and_load_data(data, args, detector, matcher)
            # Check if the dataset has a downsize factor attribute
            if args.features == "RoMA" and hasattr(dataset, "downsize_factor") and dataset.downsize_factor is not None:
                matches = matches / dataset.downsize_factor
            processing_queue.append((data, matches, scores))
            
            print(f"Processing pair [{i + 1} / {len(dataloader)}]")
                                            
            ## Running the estimators so we don't have too much things in the memory
            if len(processing_queue) >= args.batch_size or i == len(dataloader) - 1:
                for method in ["vsac", "GCRANSAC OpenCV", "GC-RANSAC github", "RANSAC OpenCV", "RHO OpenCV", "LMEDS OpenCV", "USAC_MAGSAC OpenCV", "MAGSAC github", "MAGSAC++ github", "poselib", "pycolmap"]:
                    for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                        key = (method, iters)
                        if key not in pose_errors:
                            pose_errors[key] = []
                            runtimes[key] = []
                            inlier_numbers[key] = []

                        args.inlier_threshold = thresholds[method]
                        args.method = method
                        args.maximum_iterations = iters
                        args.minimum_iterations = iters

                        results = Parallel(n_jobs=min(args.core_number, len(processing_queue)))(delayed(run)(
                            matches,
                            scores,
                            data["K1"],
                            data["K2"],
                            data["R_1_2"],
                            data["T_1_2"],
                            args) for data, matches, scores in tqdm(processing_queue))

                        # Concatenating the results to the main lists
                        pose_errors[key] += [error for error, inlier_number, time in results]
                        runtimes[key] += [time for error, inlier_number, time in results]
                        inlier_numbers[key] += [inlier_number for error, inlier_number, time in results]

                # Clearing the processing queue
                processing_queue = []
                run_count += 1

        # Write results into csv
        out = f"tests/results_testing_H_baselines_{args.features}.csv"
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write("method,features,dataset,threshold,maximum_iterations,confidence,spatial_weight,neighborhood_size,sampler,space_partitioning,sprt,auc_R5,auc_R10,auc_R20,auc_t5,auc_t10,auc_t20,auc_Rt5,auc_Rt10,auc_Rt20,avg_error,med_error,avg_time,median_time,avg_inliers,median_inliers,variance,solver,scoring\n")
        with open(out, "a") as f:
            for method in ["vsac", "GCRANSAC OpenCV", "GC-RANSAC github", "RANSAC OpenCV", "RHO OpenCV", "LMEDS OpenCV", "USAC_MAGSAC OpenCV", "MAGSAC github", "MAGSAC++ github", "poselib", "pycolmap"]:
                for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                    key = (method, iters)
                    curr_pose_errors = np.array(pose_errors[key])
                    auc_R = 100 * np.r_[pose_auc(curr_pose_errors[:,0], thresholds=[5, 10, 20])]
                    auc_t = 100 * np.r_[pose_auc(curr_pose_errors[:,1], thresholds=[5, 10, 20])]
                    auc_Rt = 100 * np.r_[pose_auc(curr_pose_errors.max(1), thresholds=[5, 10, 20])]

                    # Remove inf values
                    curr_pose_errors = curr_pose_errors[np.isfinite(curr_pose_errors).all(axis=1)]
                    f.write(f"{method},{args.features},{db_name},{thresholds[method]},{iters},{args.confidence},{args.spatial_coherence_weight},{args.neighborhood_grid_density},{args.sampler},0,0,{auc_R[0]},{auc_R[1]},{auc_R[2]},{auc_t[0]},{auc_t[1]},{auc_t[2]},{auc_Rt[0]},{auc_Rt[1]},{auc_Rt[2]},{np.mean(curr_pose_errors)},{np.median(curr_pose_errors)},{np.mean(runtimes[key])},{np.median(runtimes[key])},{np.mean(inlier_numbers[key])},{np.median(inlier_numbers[key])},0,{0},{0}\n")
