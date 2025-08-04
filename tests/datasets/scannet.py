import os
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightglue.utils import load_image

def simple_collate_fn(sample):
    return sample


class ScanNet(Dataset):
    def __init__(self, root_dir, split):
        assert split in ["test"], "Only split currently accepted is test."
        self.name = "ScanNet"
        self.root_dir = root_dir
        # Extract image pairs
        pair_file = os.path.join(root_dir, 'scannet_test_pairs_with_gt.txt')
        with open(pair_file, 'r') as f:
            pairs = [l.split() for l in f.readlines()]
        self.img_pairs, self.K1, self.K2, self.T_1_2, self.R_1_2 = [], [], [], [], []
        for p in pairs:
            self.img_pairs.append((os.path.join(root_dir, p[0]),
                                   os.path.join(root_dir, p[1])))
            self.K1.append(np.array(p[4:13]).astype(float).reshape(3, 3))
            self.K2.append(np.array(p[13:22]).astype(float).reshape(3, 3))
            T = np.array(p[22:]).astype(float).reshape(4, 4)
            self.T_1_2.append(T[:3, 3])
            self.R_1_2.append(T[:3, :3])

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
            'T_1_2': self.T_1_2[item],
            'K1': self.K1[item],
            'K2': self.K2[item],
        }

        return outputs

    def __len__(self):
        return len(self.img_pairs)


