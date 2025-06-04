import os
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightglue.utils import load_image

def simple_collate_fn(sample):
    return sample


class PhotoTourism(Dataset):
    def __init__(self, root_dir, split, scene=None, load_points=False):
        assert split in ["train", "val", "test"], "Only split accepted are train and val."
        self.root_dir = root_dir
        self.base_dir = os.path.join(root_dir, split)
        seqs = os.listdir(self.base_dir)
        self.load_points = load_points
        self.name = "PhotoTourism"

        # Extract each pair
        self.seqs, self.img_pairs = [], []
        if scene is None:
            for seq in seqs:
                if not os.path.isdir(os.path.join(self.base_dir, seq)):
                    continue
                with h5py.File(os.path.join(self.base_dir, seq, 'Fgt.h5'), 'r') as f:
                    keys = [key for key in f.keys()]
                    self.img_pairs += keys
                    self.seqs += [seq] * len(keys)
        else:
            with h5py.File(os.path.join(self.base_dir, scene, 'Fgt.h5'), 'r') as f:
                keys = [key for key in f.keys()]
                self.img_pairs += keys
                self.seqs += [scene] * len(keys)

    def get_dataloader(self):
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=True,
            num_workers=4, collate_fn=simple_collate_fn)

    def __getitem__(self, item):
        seq = self.seqs[item]
        img_pair = self.img_pairs[item]
        name1, name2 = img_pair.split('-')

        # Read the images
        img_file1 = os.path.join(self.base_dir, seq, 'images', name1 + '.jpg')
        img_file2 = os.path.join(self.base_dir, seq, 'images', name2 + '.jpg')
        img1 = load_image(img_file1)
        img2 = load_image(img_file2)

        # Read the relative translation from img 1 to img 2
        with h5py.File(os.path.join(self.base_dir, seq, 'R.h5'), 'r') as f:
            R1 = f[name1][()]
            R2 = f[name2][()]
            R_1_2 = np.dot(R2, R1.T)

        # Read the relative translation from img 1 to img 2
        with h5py.File(os.path.join(self.base_dir, seq, 'T.h5'), 'r') as f:
            T1 = f[name1][()]
            T2 = f[name2][()]
            T_1_2 = T2 - np.dot(R_1_2, T1)

        # Read the intrinsics
        with h5py.File(os.path.join(self.base_dir, seq, 'K1_K2.h5'), 'r') as f:
            K = f[img_pair][()]
            K1 = K[0][0]
            K2 = K[0][1]

        outputs = {
            'id1': seq + '/images/' + name1 + '.jpg',
            'id2': seq + '/images/' + name2 + '.jpg',
            'img1': img1,
            'img2': img2,
            'R_1_2': R_1_2,
            'T_1_2': T_1_2,
            'K1': K1,
            'K2': K2,
        }

        # Optionally load points and score (the lower the better)
        if self.load_points:
            with h5py.File(os.path.join(self.base_dir, seq, 'matches.h5'), 'r') as f:
                outputs["kp_matches"] = f[img_pair][()]
            with h5py.File(os.path.join(self.base_dir, seq, 'match_conf.h5'), 'r') as f:
                outputs["kp_scores"] = f[img_pair][()]

        return outputs

    def __len__(self):
        return len(self.img_pairs)
