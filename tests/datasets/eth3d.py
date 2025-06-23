"""
ETH3D multi-view benchmark, used for relative pose evaluation.
"""
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from lightglue.utils import load_image
import torch

def simple_collate_fn(sample):
    return sample


def read_cameras(camera_file, scale_factor=None):
    """ Read the camera intrinsics from a file in COLMAP format. """
    with open(camera_file, 'r') as f:
        raw_cameras = f.read().rstrip().split('\n')
    raw_cameras = raw_cameras[3:]
    cameras = []
    for c in raw_cameras:
        data = c.split(' ')
        cameras.append({
            "model": data[1],
            "width": int(data[2]),
            "height": int(data[3]),
            "params": np.array(list(map(float, data[4:])))})

    # Optionally scale the intrinsics if the image are resized
    if scale_factor is not None:
        cameras = [scale_intrinsics(c, scale_factor) for c in cameras]
    return cameras


def scale_intrinsics(intrinsics, scale_factor):
    """ Adapt the camera intrinsics to an image resize. """
    new_intrinsics = {"model": intrinsics["model"],
                      "width": int(intrinsics["width"] * scale_factor + 0.5),
                      "height": int(intrinsics["height"] * scale_factor + 0.5)
                      }
    params = intrinsics["params"]
    # Adapt the focal length
    params[:2] *= scale_factor
    # Adapt the principal point
    params[2:4] = (params[2:4] * scale_factor + 0.5) - 0.5
    new_intrinsics["params"] = params
    return new_intrinsics


def intrinsics2K(intrinsics):
    """ Extract the K matrix from the COLMAP pinhole model. """
    params = intrinsics["params"]
    return np.array([[params[0], 0., params[2]],
                     [0., params[1], params[3]],
                     [0., 0., 1.]])


def qvec2rotmat(qvec):
    """ Convert from quaternions to rotation matrix. """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


class ETH3D(Dataset):
    # Initialize the dataset
    def __init__(self, root_dir, split="test", downsize_factor=8,
                 min_covisibility=500):
        if split != "test":
            raise ValueError("[Error] ETH3D only available in 'test' mode.")
        self.downsize_factor = downsize_factor
        self.name = "ETH3D"

        # Form pairs of images from the multiview dataset
        self.root_dir = root_dir
        self.img_dir = Path(root_dir)
        self.data = []
        for folder in self.img_dir.iterdir():
            img_folder = Path(folder, "images", "dslr_images_undistorted")
            # Check whether img_folder is a directory
            if not img_folder.is_dir():
                continue
            names = [img.name for img in img_folder.iterdir()]
            names.sort()

            # Read intrinsics and extrinsics data
            undist_cameras = read_cameras(str(
                Path(folder, "dslr_calibration_undistorted", "cameras.txt")),
                1 / self.downsize_factor)
            name_to_cam_idx = {name: {} for name in names}
            with open(str(Path(folder, "dslr_calibration_jpg", "images.txt")),
                      "r") as f:
                raw_data = f.read().rstrip().split('\n')[4::2]
            for raw_line in raw_data:
                line = raw_line.split(' ')
                img_name = os.path.basename(line[-1])
                name_to_cam_idx[img_name]["dist_camera_idx"] = int(line[-2])
            T_world_to_camera = {}
            image_visible_points3D = {}
            with open(str(Path(folder, "dslr_calibration_undistorted",
                               "images.txt")), "r") as f:
                lines = f.readlines()[4 :]  # Skip the header
                raw_poses = [line.strip('\n').split(' ')
                             for line in lines[:: 2]]
                raw_points = [line.strip('\n').split(' ')
                              for line in lines[1 :: 2]]
            for raw_pose, raw_pts in zip(raw_poses, raw_points):
                img_name = os.path.basename(raw_pose[-1])
                # Extract the transform from world to camera
                target_extrinsics = list(map(float, raw_pose[1:8]))
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = qvec2rotmat(target_extrinsics[:4])
                pose[:3, 3] = target_extrinsics[4:]
                T_world_to_camera[img_name] = pose
                name_to_cam_idx[img_name]["undist_camera_idx"] = int(raw_pose[-2])
                # Extract the visible 3D points
                point3D_ids = [id for id in map(int, raw_pts[2 :: 3])
                               if id != -1]
                image_visible_points3D[img_name] = set(point3D_ids)
            
            # Extract the covisibility of each image
            num_imgs = len(names)
            n_covisible_points = np.zeros((num_imgs, num_imgs))
            for i in range(num_imgs - 1):
                for j in range(i + 1, num_imgs):
                    visible_points3D1 = image_visible_points3D[names[i]]
                    visible_points3D2 = image_visible_points3D[names[j]]
                    n_covisible_points[i, j] = len(visible_points3D1 & visible_points3D2)

            # Keep only the pairs with enough covisibility
            valid_pairs = np.where(n_covisible_points >= min_covisibility)
            valid_pairs = np.stack(valid_pairs, axis=1)
            
            # Extract the paths
            self.data += [{
                "ref_name": names[i][:-4],
                "target_name": names[j][:-4],
                "ref_img_path": str(Path(img_folder, names[i])),
                "target_img_path": str(Path(img_folder, names[j])),
                "ref_undist_camera": intrinsics2K(
                    undist_cameras[name_to_cam_idx[names[i]]["undist_camera_idx"]]),
                "target_undist_camera": intrinsics2K(
                    undist_cameras[name_to_cam_idx[names[j]]["undist_camera_idx"]]),
                "T_world_to_ref": T_world_to_camera[names[i]],
                "T_world_to_target": T_world_to_camera[names[j]],
                "n_covisible_points": n_covisible_points[i, j]}
                for (i, j) in valid_pairs]

        print(f"Initialized ETH3D dataset with {len(self.data)} image pairs.")

    def get_dataloader(self):
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=True,
            num_workers=4, collate_fn=simple_collate_fn)

    def __getitem__(self, idx):
        # Load the reference image
        data = self.data[idx]
        ref_img = cv2.imread(data["ref_img_path"])
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        output_shape = np.array(ref_img.shape) // self.downsize_factor
        ref_img = cv2.resize(ref_img, tuple(output_shape[[1, 0]]),
                             interpolation = cv2.INTER_AREA)
        # Convert to torch and move the last channel to the front
        ref_img = torch.from_numpy(ref_img.transpose(2, 0, 1)).float() / 255.0

        # Load the target image
        target_img = cv2.imread(data["target_img_path"])
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        output_shape = np.array(target_img.shape) // self.downsize_factor
        target_img = cv2.resize(target_img, tuple(output_shape[[1, 0]]),
                                interpolation = cv2.INTER_AREA)
        # Convert to torch and move the last channel to the front
        target_img = torch.from_numpy(target_img.transpose(2, 0, 1)).float() / 255.0

        # Get the relative pose
        Rt = data["T_world_to_target"] @ np.linalg.inv(data["T_world_to_ref"])

        return {
            "id1": data["ref_name"], "id2": data["target_name"],
            "path1": data["ref_img_path"], "path2": data["target_img_path"],
            "img1": ref_img, "img2": target_img,
            "R_1_2": Rt[:3, :3], "T_1_2": Rt[:3, 3],
            "K1": data["ref_undist_camera"],
            "K2": data["target_undist_camera"],
            "n_covisible_points": data["n_covisible_points"]
        }

    def __len__(self):
        return len(self.data)
