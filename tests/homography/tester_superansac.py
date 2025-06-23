import os
import numpy as np
import argparse
import sys
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint
from utils import detect_and_load_data
from romatch import roma_outdoor
from errors import homography_pose_error

# Add the parent directory (../) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from evaluation import pose_auc

from datasets.scannet import ScanNet
from datasets.lamar import Lamar
from datasets.eth3d import ETH3D
from datasets.kitti import Kitti
from datasets.phototourism import PhotoTourism
from datasets.seven_scenes import SevenScenes

import pysuperansac

def run(matches, scores, K1, K2, R_gt, t_gt, image_size1, image_size2, args):
    # Sort the matches by matching score
    indices = np.argsort(scores)[::-1]
    matches = matches[indices, :]
    scores = scores[indices]

    # Return if there are fewer than 4 correspondences
    if matches.shape[0] < 8:
        return (np.inf, np.inf), 0, 0
    
    # Set up the configuration
    config = pysuperansac.RANSACSettings()
    config.inlier_threshold = args.inlier_threshold
    config.min_iterations = args.minimum_iterations
    config.max_iterations = args.maximum_iterations
    config.confidence = args.confidence
    if args.sampler == "Uniform":
        config.sampler = pysuperansac.SamplerType.Uniform
    elif args.sampler == "PROSAC":
        config.sampler = pysuperansac.SamplerType.PROSAC
    elif args.sampler == "PNAPSAC":
        config.sampler = pysuperansac.SamplerType.ProgressiveNAPSAC
    elif args.sampler == "Importance":
        config.sampler = pysuperansac.SamplerType.ImportanceSampler
        
    if args.scoring == "RANSAC":
        config.scoring = pysuperansac.ScoringType.RANSAC
    elif args.scoring == "MSAC":
        config.scoring = pysuperansac.ScoringType.MSAC
    elif args.scoring == "MAGSAC":
        config.scoring = pysuperansac.ScoringType.MAGSAC
    elif args.scoring == "ACRANSAC":
        config.scoring = pysuperansac.ScoringType.ACRANSAC
    elif args.scoring == "GaU":
        config.scoring = pysuperansac.ScoringType.GAU
    elif args.scoring == "ML":
        config.scoring = pysuperansac.ScoringType.ML
        
    if args.lo == "LSQ":
        config.local_optimization = pysuperansac.LocalOptimizationType.LSQ
    elif args.lo == "IRLS":
        config.local_optimization = pysuperansac.LocalOptimizationType.IteratedLSQ
    elif args.lo == "NestedRANSAC":
        config.local_optimization = pysuperansac.LocalOptimizationType.NestedRANSAC
    elif args.lo == "GCRANSAC":
        config.local_optimization = pysuperansac.LocalOptimizationType.GCRANSAC
    elif args.lo == "IteratedLMEDS":
        config.local_optimization = pysuperansac.LocalOptimizationType.IteratedLMEDS
    elif args.lo == "Nothing":
        config.local_optimization = pysuperansac.LocalOptimizationType.Nothing
        
    if args.fo == "LSQ":
        config.final_optimization = pysuperansac.LocalOptimizationType.LSQ
    elif args.fo == "IRLS":
        config.final_optimization = pysuperansac.LocalOptimizationType.IteratedLSQ
    elif args.fo == "NestedRANSAC":
        config.final_optimization = pysuperansac.LocalOptimizationType.NestedRANSAC
    elif args.fo == "GCRANSAC":
        config.local_optimization = pysuperansac.LocalOptimizationType.GCRANSAC
    elif args.fo == "IteratedLMEDS":
        config.local_optimization = pysuperansac.LocalOptimizationType.IteratedLMEDS
    elif args.fo == "Nothing":
        config.final_optimization = pysuperansac.LocalOptimizationType.Nothing
        
    config.neighborhood_settings.neighborhood_grid_density = 6
    config.neighborhood_settings.neighborhood_size = args.neighborhood_size
    
    # If Importance sampler or ARSampler is used, the SNN ratios are converted as probabilities
    probabilities = []
    if args.sampler == "Importance" or args.sampler == "ARSampler":
        point_number = matches.shape[0]
        for i in range(point_number):
            probabilities.append(1.0 - i / point_number)

    # Run the homography estimation implemented in OpenCV
    tic = time.perf_counter()
    H_est, inliers, score, iterations = pysuperansac.estimateHomography(
        np.ascontiguousarray(matches), 
        probabilities,
        [image_size1[2], image_size1[1], image_size2[2], image_size2[1]],
        config = config)
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
    parser = argparse.ArgumentParser(description="Running homography estimation with SupeRANSAC")
    parser.add_argument('--features', type=str, help="Choose from: SP+LG, RoMA.", choices=["splg", "RoMA"], default="splg")
    parser.add_argument('--batch_size', type=int, help="Batch size for multi-CPU processing", default=2000)
    parser.add_argument("--confidence", type=float, default=0.9999999)
    parser.add_argument("--inlier_threshold", type=float, default=-1)
    parser.add_argument("--minimum_iterations", type=int, default=1000)
    parser.add_argument("--maximum_iterations", type=int, default=1000)
    parser.add_argument("--sampler", type=str, help="Choose from: Uniform, PROSAC, PNAPSAC, Importance, ARSampler.", choices=["Uniform", "PROSAC", "PNAPSAC", "Importance", "ARSampler"], default="PROSAC")
    parser.add_argument("--scoring", type=str, help="Choose from: RANSAC, MSAC, MAGSAC, ACRANSAC.", choices=["RANSAC", "MSAC", "MAGSAC", "ACRANSAC"], default="MAGSAC")
    parser.add_argument("--lo", type=str, help="Choose from: LSQ, IRLS, NestedRANSAC, Nothing.", choices=["LSQ", "IRLS", "NestedRANSAC", "GCRANSAC", "IteratedLMEDS", "Nothing"], default="GCRANSAC")
    parser.add_argument("--fo", type=str, help="Choose from: LSQ, IRLS, NestedRANSAC, Nothing.", choices=["LSQ", "IRLS", "NestedRANSAC", "GCRANSAC", "IteratedLMEDS", "Nothing"], default="IRLS")
    parser.add_argument("--spatial_coherence_weight", type=float, default=0.8)
    parser.add_argument("--neighborhood_size", type=float, default=20)
    parser.add_argument("--neighborhood_grid_density", type=float, default=5)
    parser.add_argument("--core_number", type=int, default=18)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    out = f"tests/homography/results_testing_superansac_{args.features}.csv"
    
    print(f"Testing SupeRANSAC")

    if args.features == "splg":
        print("Initialize SP+LG detector")
        detector = SuperPoint(max_num_keypoints=2048).eval().to(args.device)  # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(args.device)  # load the matcher
        if args.inlier_threshold <= 0:
            args.inlier_threshold = 5.0
            print(f"Setting the threshold to {args.inlier_threshold} px as it works best for H estimation with SP+LG features.")
    elif args.features == "RoMA":
        print("Initialize RoMA detector")
        detector = roma_outdoor(device = args.device)
        matcher = None
        if args.lo == "GCRANSAC":
            args.lo = "NestedRANSAC"
        if args.inlier_threshold <= 0:
            args.inlier_threshold = 2.0
            print(f"Setting the threshold to {args.inlier_threshold} px as it works best for H estimation with RoMA features.")
    
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
            
            if (i + 1) % 100 == 0:
                print(f"Processing pair [{i + 1} / {len(dataloader)}]")
            
            ## Running the estimators so we don't have too much things in the memory
            if len(processing_queue) >= args.batch_size or i == len(dataloader) - 1:
                for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                    key = iters
                    if iters not in pose_errors:
                        pose_errors[key] = []
                        runtimes[key] = []
                        inlier_numbers[key] = []

                    args.maximum_iterations = iters
                    args.minimum_iterations = iters

                    results = Parallel(n_jobs=min(args.core_number, len(processing_queue)))(delayed(run)(
                        matches,
                        scores,
                        data["K1"],
                        data["K2"],
                        data["R_1_2"],
                        data["T_1_2"],
                        data["img1"].shape,
                        data["img2"].shape,
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

        # Write results into csv
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write("method,features,dataset,threshold,maximum_iterations,confidence,spatial_weight,neighborhood_size,sampler,space_partitioning,sprt,auc_R5,auc_R10,auc_R20,auc_t5,auc_t10,auc_t20,auc_Rt5,auc_Rt10,auc_Rt20,avg_error,med_error,avg_time,median_time,avg_inliers,median_inliers,variance,solver,scoring\n")
        with open(out, "a") as f:
            for iters in [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]:
                key = iters
                curr_pose_errors = np.array(pose_errors[key])
                auc_R = 100 * np.r_[pose_auc(curr_pose_errors[:,0], thresholds=[5, 10, 20])]
                auc_t = 100 * np.r_[pose_auc(curr_pose_errors[:,1], thresholds=[5, 10, 20])]
                auc_Rt = 100 * np.r_[pose_auc(curr_pose_errors.max(1), thresholds=[5, 10, 20])]

                # Remove inf values
                curr_pose_errors = curr_pose_errors[np.isfinite(curr_pose_errors).all(axis=1)]
                f.write(f"superansac,{args.features},{db_name},{args.inlier_threshold},{iters},{args.confidence},{args.spatial_coherence_weight},{args.neighborhood_grid_density},{args.sampler},0,0,{auc_R[0]},{auc_R[1]},{auc_R[2]},{auc_t[0]},{auc_t[1]},{auc_t[2]},{auc_Rt[0]},{auc_Rt[1]},{auc_Rt[2]},{np.mean(curr_pose_errors)},{np.median(curr_pose_errors)},{np.mean(runtimes[key])},{np.median(runtimes[key])},{np.mean(inlier_numbers[key])},{np.median(inlier_numbers[key])},0,{0},{0}\n")

