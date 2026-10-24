import os
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

def simple_collate_fn(sample):
    return sample


class PhotoTourism(Dataset):
    def __init__(self, root_dir, split="val", scene=None, load_points=False, max_pairs=150):
        assert split in ["train", "val", "test"], "Only split accepted are train, val, and test."
        self.root_dir = root_dir
        self.base_dir = os.path.join(root_dir, split)
        seqs = sorted(os.listdir(self.base_dir)) if os.path.isdir(self.base_dir) else []
        self.load_points = load_points
        self.name = "PhotoTourism"

        # Extract each pair whose images exist on disk
        self.seqs, self.img_pairs = [], []
        target_seqs = seqs if scene is None else [scene]
        for seq in target_seqs:
            seq_dir = os.path.join(self.base_dir, seq)
            fgt_path = os.path.join(seq_dir, "Fgt.h5")
            if not os.path.isdir(seq_dir) or not os.path.exists(fgt_path):
                continue
            with h5py.File(fgt_path, "r") as f:
                keys = sorted(list(f.keys()))
            valid_keys = []
            for k in keys:
                n1, n2 = k.split("-")
                p1 = os.path.join(seq_dir, "images", n1 + ".jpg")
                p2 = os.path.join(seq_dir, "images", n2 + ".jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    valid_keys.append(k)
                    if max_pairs and len(self.img_pairs) + len(valid_keys) >= max_pairs:
                        break
            self.img_pairs += valid_keys
            self.seqs += [seq] * len(valid_keys)
            if max_pairs and len(self.img_pairs) >= max_pairs:
                break

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
            'path1': img_file1,
            'path2': img_file2,
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
