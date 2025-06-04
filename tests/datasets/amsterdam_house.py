import os
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightglue.utils import load_image

f = 2902.0083461890194
K = np.array([[f, 0, 800],
              [0, f, 600],
              [0, 0, 1]])


def simple_collate_fn(sample):
    return sample


class AmsterdamHouse(Dataset):
    def __init__(self, root_dir, split, use_mean_intrinsics=True):
        assert split in ["test"], "Only split currently accepted is test."

        (self.img_pairs, self.T_1_2, self.R_1_2,
         self.K1, self.K2) = [], [], [], [], []
        img_paths = [os.path.join(root_dir, p)
                     for p in os.listdir(root_dir) if p[-3:] == 'jpg']
        img_paths.sort()
        proj_paths = [os.path.join(root_dir, p)
                     for p in os.listdir(root_dir) if p[-10:] == 'projmatrix']
        proj_paths.sort()

        for i in range(len(img_paths) - 1):
            self.img_pairs.append((img_paths[i], img_paths[i + 1]))
            P1 = np.loadtxt(proj_paths[i])
            P2 = np.loadtxt(proj_paths[i + 1])
            K1, R1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
            K2, R2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
            t1, t2 = t1[:3, 0] / t1[3, 0], t2[:3, 0] / t2[3, 0]
            t1, t2 = -R1.dot(t1), -R2.dot(t2)
            if use_mean_intrinsics:
                self.K1.append(K)
                self.K2.append(K)
            else:
                self.K1.append(K1)
                self.K2.append(K2)
            self.R_1_2.append(R2.dot(R1.T))
            self.T_1_2.append(t2 - R2.dot(R1.T).dot(t1))

        print(f"Initialized Amsterdam House dataset with {len(self.img_pairs)} image pairs.")

    def get_dataloader(self):
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=True,
            num_workers=4, collate_fn=simple_collate_fn)

    def __getitem__(self, item):
        # Read the images
        img1 = cv2.cvtColor(cv2.imread(self.img_pairs[item][0]),
                            cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(self.img_pairs[item][1]),
                            cv2.COLOR_BGR2RGB)

        outputs = {
            'id1': self.img_pairs[item][0],
            'id2': self.img_pairs[item][1],
            'img1': img1,
            'img2': img2,
            'R_1_2': self.R_1_2[item],
            'T_1_2': self.T_1_2[item],
            'K1': self.K1[item],
            'K2': self.K2[item],
        }

        return outputs

    def __len__(self):
        return len(self.img_pairs)
