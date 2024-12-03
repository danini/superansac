import os
import cv2
import numpy as np
import pykitti
from pykitti.utils import read_calib_file
from torch.utils.data import Dataset, DataLoader
from lightglue.utils import load_image

def simple_collate_fn(sample):
    return sample


class Kitti(Dataset):
    def __init__(self, root_dir, sequences='all', steps=20):
        self.root_dir = root_dir
        self.name = "KITTI"
        if sequences == 'all':
            sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08',
                         '09', '10']
        elif isinstance(sequences, str):
            sequences = [sequences]
        else:
            assert isinstance(sequences, list), "sequences must be a str or list."
        # Use the KITTI data loader for each sequence
        (self.img_pairs, self.K1, self.K2, self.T_1_2, self.R_1_2,
         self.R1, self.R2) = [], [], [], [], [], [], []
        for s in sequences:
            data = pykitti.odometry(root_dir, s, steps=steps)
            num_imgs = len(data.cam0_files)
            # Load the image pairs
            self.img_pairs += [(data.cam0_files[i], data.cam0_files[i + 1])
                               for i in range(num_imgs - 1)]
            # Load the calibration data
            calib_filepath = os.path.join(data.sequence_path, 'calib.txt')
            filedata = read_calib_file(calib_filepath)
            K = np.reshape(filedata['P0'], (3, 4))[:3, :3]
            self.K1 += [K] * num_imgs
            self.K2 += [K] * num_imgs
            # Load the poses
            for i in range(num_imgs - 1):
                T_1_w = data.poses[i]
                T_2_w = data.poses[i + 1]
                self.R1.append(T_1_w[:3, :3].T)
                self.R2.append(T_2_w[:3, :3].T)
                T_1_2 = np.linalg.inv(T_2_w) @ T_1_w
                self.T_1_2.append(T_1_2[:3, 3])
                self.R_1_2.append(T_1_2[:3, :3])

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
            'img1': img1,
            'img2': img2,
            'R_1_2': self.R_1_2[item],
            'T_1_2': self.T_1_2[item],
            'R_world_to_cam1': self.R1[item],
            'R_world_to_cam2': self.R2[item],
            'K1': self.K1[item],
            'K2': self.K2[item],
        }

        return outputs

    def __len__(self):
        return len(self.img_pairs)
