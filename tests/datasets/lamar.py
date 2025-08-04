import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightglue.utils import load_image

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

def simple_collate_fn(sample):
    return sample


calib = {
    'hetlf': np.array([[363.32, 0, 235.031], [0, 360.299, 288.286], [0, 0, 1]]),
    'hetrf': np.array([[364.494, 0, 242.239], [0, 362.158, 297.774], [0, 0, 1]]),
    'hetrr': np.array([[365.116, 0, 245.76], [0, 363.651, 299.386], [0, 0, 1]]),
    'hetll': np.array([[363.605, 0, 246.758], [0, 362.364, 303.493], [0, 0, 1]]),
}


def transform_vec(T, v):
    """ Transform vector v in 3D with T in 4x4. """
    v_t = T @ np.r_[v, 1.]
    return v_t[:3] / v_t[3]


class Lamar(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.name = "lamar"
        # Extract the mapping img name -> rig ID
        img2rig_id = {}
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
            for l in lines:
                img2rig_id[l[2][-13:]] = l[0]

        # Extract the GT transformation from world to rig
        T_w_rig = {}
        with open(os.path.join(root_dir, 'proc/alignment_trajectories.txt'), 'r') as f:
            lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
            for l in lines:
                R_rig_w = qvec2rotmat(
                    np.array(l[2:6]).astype(np.float32))
                t_rig_w = np.array(l[6:9], dtype=np.float32)
                T_w_rig[l[0]] = np.eye(4)
                T_w_rig[l[0]][:3, :3] = R_rig_w.T
                T_w_rig[l[0]][:3, 3] = -R_rig_w.T @ t_rig_w

        # Extract the transformation from rig to cam
        T_rig_cam = {}
        with open(os.path.join(root_dir, 'rigs.txt'), 'r') as f:
            lines = [s.strip('\n').split(', ') for s in f.readlines()][1:]
            for l in lines:
                rig_id = l[0][-9:]
                if rig_id not in T_rig_cam:
                    T_rig_cam[rig_id] = {}
                R_cam_rig = qvec2rotmat(np.array(l[2:6]).astype(np.float32))
                t_cam_rig = np.array(l[6:9], dtype=np.float32)
                T_rig_cam[rig_id][l[1][-5:]] = np.eye(4)
                T_rig_cam[rig_id][l[1][-5:]][:3, :3] = R_cam_rig.T
                T_rig_cam[rig_id][l[1][-5:]][:3, 3] = -R_cam_rig.T @ t_cam_rig

        (self.img_pairs, self.K1, self.K2, self.R_1_2, self.T_1_2, self.R1, self.R2) = [], [], [], [], [], [], []
        sequences = os.listdir(os.path.join(root_dir, 'raw_data'))
        for s in sequences:
            seq_path = os.path.join(root_dir, 'raw_data', s)

            # Extract the images
            for cam in ['hetlf', 'hetll', 'hetrf', 'hetrr']:
                img_folder = os.path.join(seq_path, 'images', cam)
                images = os.listdir(img_folder)
                images.sort()
                for i in range(len(images) - 1):
                    rig_id = img2rig_id[images[i]]
                    next_rig_id = img2rig_id[images[i + 1]]
                    self.img_pairs.append((
                        os.path.join(img_folder, images[i]),
                        os.path.join(img_folder, images[i + 1])))
                    T_w_1 = T_rig_cam[rig_id][cam] @ T_w_rig[rig_id]
                    self.R1.append(T_w_1[:3, :3])
                    T_w_2 = T_rig_cam[next_rig_id][cam] @ T_w_rig[next_rig_id]
                    self.R2.append(T_w_2[:3, :3])
                    self.R_1_2.append(self.R2[-1] @ self.R1[-1].T)
                    self.T_1_2.append(T_w_2 @ np.linalg.inv(T_w_1))
                    self.K1.append(calib[cam])
                    self.K2.append(calib[cam])

    def get_dataloader(self):
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=True,
            num_workers=4, collate_fn=simple_collate_fn)

    def __getitem__(self, item):
        # Read the images
        img1 = load_image(self.img_pairs[item][0])
        img2 = load_image(self.img_pairs[item][1])

        outputs = {
            'id1': self.img_pairs[item][0],
            'id2': self.img_pairs[item][1],
            'path1': self.img_pairs[item][0],
            'path2': self.img_pairs[item][1],
            'img1': img1,
            'img2': img2,
            'R_1_2': self.R_1_2[item],
            'T_1_2': self.T_1_2[item][:3, 3],
            'R_world_to_cam1': self.R1[item],
            'R_world_to_cam2': self.R2[item],
            'K1': self.K1[item],
            'K2': self.K2[item],
        }

        return outputs

    def __len__(self):
        return len(self.img_pairs)
