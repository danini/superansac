import cv2
import os
import numpy as np
import argparse
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from romatch import roma_outdoor
import sys
import torch

# Add the parent directory (../) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from functions import normalize_keypoints
from evaluation import evaluate_R_t, pose_auc
from utils import detect_and_load_data
from evaluation import evaluate_R_t, pose_auc

import pygcransac
import pymagsac
import poselib
import pycolmap
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

from datasets.scannet import ScanNet
from datasets.lamar import Lamar
from datasets.eth3d import ETH3D
from datasets.kitti import Kitti
from datasets.phototourism import PhotoTourism
from datasets.seven_scenes import SevenScenes

def run(matches, scores, K1, K2, R_gt, t_gt, args):
    # Initialize the errors
    error = 1e10
    rotation_error = 1e10
    translation_error = 1e10
    absolute_translation_error = 1e10
    
    # Sort the matches by matching score
    indices = np.argsort(scores)[::-1]
    matches = matches[indices, :]
    scores = scores[indices]
    F_est = None

    # Return if there are fewer than 4 correspondences
    if matches.shape[0] < 8:
        return (np.inf, np.inf), 0, 0

    if args.method == "poselib":
        ransac_options = {"max_iterations": args.maximum_iterations,
                        "min_iterations":  args.minimum_iterations,
                        "success_prob": args.confidence,
                        "max_epipolar_error": args.inlier_threshold,
                        "progressive_sampling": True }
        
        tic = time.perf_counter()
        F_est, res  = poselib.estimate_fundamental(matches[:,:2], matches[:,2:4], ransac_opt=ransac_options)
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
        res  = pycolmap.fundamental_matrix_estimation(xy1, xy2, estimation_options = ransac_options)
        F_est = res['F']
        inliers = np.array(res['inliers'])
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "RANSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        F_est, inliers = cv2.findFundamentalMat(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.RANSAC,
            ransacReprojThreshold = args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "LMEDS OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        F_est, inliers = cv2.findFundamentalMat(
            np.ascontiguousarray(matches[:, :2]),
            np.ascontiguousarray(matches[:, 2:4]),
            cv2.LMEDS,
            ransacReprojThreshold = args.inlier_threshold,
            maxIters = args.maximum_iterations,
            confidence = args.confidence)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "skimage":
        if matches.shape[0] < 9:
            return (np.inf, np.inf), 0, 0
        tic = time.perf_counter()
        model, inliers = ransac(
            (matches[:, :2], matches[:, 2:4]),
            FundamentalMatrixTransform,
            min_samples = 8,
            residual_threshold = args.inlier_threshold,
            max_trials = args.maximum_iterations,
        )
        F_est = model.params
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "gcransac":        
        tic = time.perf_counter()
        F_est, inliers = pygcransac.findFundamentalMatrix(
            np.ascontiguousarray(matches),
            int(K1[1, 2] * 2),
            int(K1[0, 2] * 2),
            int(K2[1, 2] * 2),
            int(K2[0, 2] * 2),
            threshold = args.inlier_threshold,
            sampler = 1,
            max_iters = args.maximum_iterations,
            min_iters = args.maximum_iterations,
            probabilities=[],
        )
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "magsac":    
        tic = time.perf_counter()
        F_est, inliers = pymagsac.findFundamentalMatrix(
            np.ascontiguousarray(matches), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            max_iters = args.maximum_iterations,
            min_iters = args.maximum_iterations,
            use_magsac_plus_plus = False,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "magsac++":   
        tic = time.perf_counter()
        F_est, inliers = pymagsac.findFundamentalMatrix(
            np.ascontiguousarray(matches), 
            int(K1[0, 2] * 2),
            int(K1[1, 2] * 2),
            int(K2[0, 2] * 2),
            int(K2[1, 2] * 2),
            probabilities = [],
            sampler = 1,
            max_iters = args.maximum_iterations,
            min_iters = args.maximum_iterations,
            use_magsac_plus_plus = True,
            sigma_th = args.inlier_threshold)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "vsac":
        import pvsac

        tic = time.perf_counter()
        params = pvsac.Params(pvsac.EstimationMethod.Fundamental, args.inlier_threshold, 0.999999999999999999999999999999999999999, args.maximum_iterations, pvsac.SamplingMethod.SAMPLING_UNIFORM, pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        F_est, inliers = pvsac.estimate(params, matches[:,:2], matches[:,2:], None, None, None, None)
        toc = time.perf_counter()
        elapsed_time = toc - tic

        if np.sum(inliers) < 8:
            return (np.inf, np.inf), 0, 0
    elif args.method == "MAGSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        try:
            F_est, inliers = cv2.findFundamentalMat(
                np.ascontiguousarray(matches[:, :2]),
                np.ascontiguousarray(matches[:, 2:4]),
                cv2.USAC_MAGSAC,
                ransacReprojThreshold = args.inlier_threshold,
                maxIters = args.maximum_iterations,
                confidence = args.confidence)
        except:
            F_est = None
            inliers = np.zeros(matches.shape[0], dtype=bool)
        toc = time.perf_counter()
        elapsed_time = toc - tic
    elif args.method == "GCRANSAC OpenCV":
        # Run the homography estimation implemented in OpenCV
        tic = time.perf_counter()
        try:
            F_est, inliers = cv2.findFundamentalMat(
                np.ascontiguousarray(matches[:, :2]),
                np.ascontiguousarray(matches[:, 2:4]),
                cv2.USAC_ACCURATE,
                ransacReprojThreshold = args.inlier_threshold,
                maxIters = args.maximum_iterations,
                confidence = args.confidence)
        except:
            F_est = None
            inliers = np.zeros(matches.shape[0], dtype=bool)
        toc = time.perf_counter()
        elapsed_time = toc - tic

    if F_est is None:
        return (np.inf, np.inf), 0, elapsed_time

    # Convert the fundamental matrix to essential matrix if the estimation is successful
    E_est = K2.T @ F_est @ K1

    norm_matches = np.zeros(matches.shape)
    norm_matches[:, :2] = normalize_keypoints(matches[:, :2], K1)
    norm_matches[:, 2:] = normalize_keypoints(matches[:, 2:], K2)

    # Decompose the essential matrix to get the relative pose
    _, R, t, _ = cv2.recoverPose(E_est, norm_matches[inliers, :2], norm_matches[inliers, 2:])
    
    # Count the inliers
    inlier_number = inliers.sum()
    
    return evaluate_R_t(R_gt, t_gt, R, t), inlier_number, elapsed_time

if __name__ == "__main__":
    # Passing the arguments
    parser = argparse.ArgumentParser(description="Running on the HEB benchmark")
    parser.add_argument('--features', type=str, help="Choose from: splg, RoMA.", choices=["splg", "RoMA"], default="splg")
    parser.add_argument('--batch_size', type=int, help="Batch size for multi-CPU processing", default=1000)
    parser.add_argument('--output_db_path', type=str, help="The path to where the dataset of matches should be saved.", default="/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test/matches.h5")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--inlier_threshold", type=float, default=1.5)
    parser.add_argument("--minimum_iterations", type=int, default=1000)
    parser.add_argument("--maximum_iterations", type=int, default=1000)
    parser.add_argument("--sampler", type=str, help="Choose from: Uniform, PROSAC, PNAPSAC, Importance, ARSampler.", choices=["Uniform", "PROSAC", "PNAPSAC", "Importance", "ARSampler"], default="PROSAC")
    parser.add_argument("--spatial_coherence_weight", type=float, default=0.1)
    parser.add_argument("--neighborhood_size", type=float, default=20)
    parser.add_argument("--neighborhood_grid_density", type=float, default=4)
    parser.add_argument("--method", type=str, help="Choose from: poselib, pycolmap.", choices=["poselib", "pycolmap", "RANSAC OpenCV", "skimage", "LMEDS OpenCV", "pydegensac github", "kornia", "gcransac", "magsac", "magsac++", "vsac"], default="vsac")
    parser.add_argument("--core_number", type=int, default=18)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print(f"Testing {args.method}")
    
    if args.features == "splg":
        print("Initialize SP+LG detector")
        detector = SuperPoint(max_num_keypoints=2048).eval().to(args.device)  # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(args.device)  # load the matcher
        
        thresholds = {
            "LMEDS OpenCV": 1.0,
            "RANSAC OpenCV": 0.75,
            "gcransac": 0.75,
            "magsac": 1.5,
            "magsac++": 3.0,
            "poselib": 2.0,
            "pycolmap": 0.5,
            "vsac": 1.5,
            "GCRANSAC OpenCV": 1.5,
            "MAGSAC OpenCV": 0.5
        }
    elif args.features == "RoMA":
        print("Initialize RoMA detector")
        detector = roma_outdoor(device = args.device)
        matcher = None
        
        thresholds = {
            "LMEDS OpenCV": 1.0,
            "RANSAC OpenCV": 1.0,
            "gcransac": 0.75,
            "magsac": 1.0,
            "magsac++": 1.5,
            "poselib": 0.75,
            "pycolmap": 0.5,
            "vsac": 0.5,
            "GCRANSAC OpenCV": 3.0,
            "MAGSAC OpenCV": 0.5
        }

    dataset_paths = ["/media/hdd3tb/datasets/scannet/scannet_lines_project/ScanNet_test", 
                     "/media/hdd2tb/datasets/RANSAC-Tutorial-Data",
                     "/media/hdd3tb/datasets/lamar/CAB/sessions/query_val_hololens",
                      "/media/hdd2tb/datasets/7scenes",
                     "/media/hdd3tb/datasets/kitti/dataset",
                     "/media/hdd3tb/datasets/eth3d"]
    datasets = [ScanNet, PhotoTourism, Lamar, SevenScenes, Kitti, ETH3D] 
    
    for idx, dataset_class in enumerate(datasets):
        if dataset_class == ScanNet:
            dataset = ScanNet(root_dir=os.path.expanduser(dataset_paths[idx]), split='test')
        elif dataset_class == PhotoTourism:
            dataset = PhotoTourism(root_dir=os.path.expanduser(dataset_paths[idx]), split='val')
        elif dataset_class == Lamar:
            dataset = Lamar(root_dir=os.path.expanduser(dataset_paths[idx]))
        elif dataset_class == ETH3D:
            dataset = ETH3D(root_dir=os.path.expanduser(dataset_paths[idx]), split='test', downsize_factor=8)
        elif dataset_class == SevenScenes:
            dataset = SevenScenes(root_dir=os.path.expanduser(dataset_paths[idx]), split='test', scene='all')
        elif dataset_class == Kitti:
            dataset = Kitti(root_dir=os.path.expanduser(dataset_paths[idx]))
        else:
            raise ValueError(f"Unknown dataset: '{dataset_class}'")
        print(f"Testing on {dataset.__class__.__name__}")
        pose_errors = {}
        runtimes = {}
        inlier_numbers = {}

        db_name = dataset.__class__.__name__.lower()
        dataloader = dataset.get_dataloader()
        args.output_db_path = dataset.root_dir + "/matches.h5"

        processing_queue = []
        run_count = 1
        for i, data in enumerate(dataloader):
            print(f"Processing pair [{i + 1} / {len(dataloader)}]")
            matches, scores = detect_and_load_data(data, args, detector, matcher)
            # Check if the dataset has a downsize factor attribute
            if args.features == "RoMA" and hasattr(dataset, "downsize_factor") and dataset.downsize_factor is not None:
                matches = matches / dataset.downsize_factor
            processing_queue.append((data, matches, scores))
                                            
            ## Running the estimators so we don't have too much things in the memory
            if len(processing_queue) >= args.batch_size or i == len(dataloader) - 1:
                for method in ["vsac", "MAGSAC OpenCV", "GCRANSAC OpenCV", "poselib", "RANSAC OpenCV", "LMEDS OpenCV", "magsac", "magsac++"]:
                    for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                        key = (method, iters)
                        if iters not in pose_errors:
                            pose_errors[key] = []
                            runtimes[key] = []
                            inlier_numbers[key] = []

                        args.maximum_iterations = iters
                        args.minimum_iterations = iters
                        args.method = method
                        args.inlier_threshold = thresholds[method]

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
                
                # Clean the memory and GPU memory
                del matches, scores
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        # Write results into csv
        out = f"tests/fundamental_matrix/results_testing_baselines_{args.features}.csv"
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write("method,features,dataset,threshold,maximum_iterations,confidence,spatial_weight,neighborhood_size,sampler,space_partitioning,sprt,auc_R5,auc_R10,auc_R20,auc_t5,auc_t10,auc_t20,auc_Rt5,auc_Rt10,auc_Rt20,avg_error,med_error,avg_time,median_time,avg_inliers,median_inliers,variance,solver,scoring\n")
        with open(out, "a") as f:
            for method in ["vsac", "MAGSAC OpenCV", "GCRANSAC OpenCV", "poselib", "RANSAC OpenCV", "LMEDS OpenCV", "magsac", "magsac++"]:
                for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                    key = (method, iters)
                    curr_pose_errors = np.array(pose_errors[key])
                    auc_R = 100 * np.r_[pose_auc(curr_pose_errors[:,0], thresholds=[5, 10, 20])]
                    auc_t = 100 * np.r_[pose_auc(curr_pose_errors[:,1], thresholds=[5, 10, 20])]
                    auc_Rt = 100 * np.r_[pose_auc(curr_pose_errors.max(1), thresholds=[5, 10, 20])]

                    # Remove inf values
                    curr_pose_errors = curr_pose_errors[np.isfinite(curr_pose_errors).all(axis=1)]
                    f.write(f"{method},{args.features},{db_name},{args.inlier_threshold},{iters},{args.confidence},{args.spatial_coherence_weight},{args.neighborhood_grid_density},{args.sampler},0,0,{auc_R[0]},{auc_R[1]},{auc_R[2]},{auc_t[0]},{auc_t[1]},{auc_t[2]},{auc_Rt[0]},{auc_Rt[1]},{auc_Rt[2]},{np.mean(curr_pose_errors)},{np.median(curr_pose_errors)},{np.mean(runtimes[key])},{np.median(runtimes[key])},{np.mean(inlier_numbers[key])},{np.median(inlier_numbers[key])},0,{0},{0}\n")

