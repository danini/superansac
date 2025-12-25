import numpy as np
import cv2
import torch

import kornia as K
import kornia.feature as KF
from lightglue.utils import load_image, rbd

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    # Optimized: avoid array allocations
    return (keypoints - K[[0, 1], 2]) / K[[0, 1], [0, 1]]

def find_relative_pose_from_points(kp_matches, K1, K2, kp_scores=None):
    if kp_scores is not None:
        # Select the points with lowest ratio score
        good_matches = kp_scores < 0.8
        pts1 = kp_matches[good_matches, :2]
        pts2 = kp_matches[good_matches, 2:]
    else:
        pts1 = kp_matches[:, :2]
        pts2 = kp_matches[:, 2:]

    if len(pts1) < 5:
        return None, None, None, None

    # Normalize KP
    p1n = normalize_keypoints(pts1, K1)
    p2n = normalize_keypoints(pts2, K2)

    # Find the essential matrix with OpenCV RANSAC
    E, inl_mask = cv2.findEssentialMat(p1n, p2n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None, None, None, None

    # Obtain the corresponding pose
    best_num_inliers = 0
    ret = None
    mask = np.array(inl_mask)[:, 0].astype(bool)
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, p1n, p2n, np.eye(3), 1e9, mask=inl_mask[:, 0])
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], pts1[mask], pts2[mask])
    return ret


def splg_matching(img1, img2, detector, matcher, device = "cuda", num_features = 2048):
    with torch.inference_mode():
        # extract local features
        feats0 = detector.extract(img1.to(device))  # auto-resize the image, disable with resize=None
        feats1 = detector.extract(img2.to(device))
        
        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    return np.concatenate([points0.cpu().numpy(), points1.cpu().numpy()], axis=1), matches01["scores"].cpu().numpy()


def loftr_matching(img1, img2, loftr_matcher, device):

    img1_raw = cv2.resize(img1, (640, 480))
    img2_raw = cv2.resize(img2, (640, 480))

    inputs = {
        'image0': torch.tensor(img1_raw, dtype=torch.float, device=device)[None, None] / 255.,
        'image1': torch.tensor(img2_raw, dtype=torch.float, device=device)[None, None] / 255.
    }
    with torch.no_grad():
        pred = loftr_matcher(inputs)
        pred = {k: v.cpu().numpy() for k, v in pred.items()}
    mkpts0, mkpts1 = pred['keypoints0'], pred['keypoints1']
    mkpts0[:, 0] = (img1.shape[1] / 640) * mkpts0[:, 0]
    mkpts0[:, 1] = (img1.shape[0] / 480) * mkpts0[:, 1]
    mkpts1[:, 0] = (img2.shape[1] / 640) * mkpts1[:, 0]
    mkpts1[:, 1] = (img2.shape[0] / 480) * mkpts1[:, 1]
    mconf = pred['confidence']
    return np.concatenate([mkpts0, mkpts1], axis=1), mconf

def point_matching(img1, img2, feature_type, detector, matcher, device = "cuda"):
    if feature_type == "splg":
        return splg_matching(img1, img2, detector, matcher, device = device)
    #elif matcher == "LoFTR":
    #    return loftr_matching(img1, img2, net, device)
    #elif matcher == "GS":
    #    return gs_matching(img1, img2, net)
    else:
        raise ValueError("Unknown matcher: " + matcher)
