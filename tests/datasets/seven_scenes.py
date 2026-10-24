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

train_splits = {
    'chess': ['seq-01', 'seq-02', 'seq-04', 'seq-06'],
    'fire': ['seq-01', 'seq-02'],
    'heads': ['seq-02'],
    'office': ['seq-01', 'seq-03', 'seq-04', 'seq-05', 'seq-08', 'seq-10'],
    'pumpkin': ['seq-02', 'seq-03', 'seq-06', 'seq-08'],
    'redkitchen': ['seq-01', 'seq-02', 'seq-05', 'seq-07', 'seq-08', 'seq-11', 'seq-13'],
    'stairs': ['seq-02', 'seq-03', 'seq-05', 'seq-06'],
}
test_splits = {
    'chess': ['seq-03', 'seq-05'],
    'fire': ['seq-03', 'seq-04'],
    'heads': ['seq-01'],
    'office': ['seq-02', 'seq-06', 'seq-07', 'seq-09'],
    'pumpkin': ['seq-01', 'seq-07'],
    'redkitchen': ['seq-03', 'seq-04', 'seq-06', 'seq-12', 'seq-14'],
    'stairs': ['seq-01', 'seq-04'],
}


def simple_collate_fn(sample):
    return sample


class SevenScenes(Dataset):
    def __init__(self, root_dir, split, scene='all'):
        assert split in ["train", "test"], "Only split currently accepted is train and test."
        self.name = "7Scenes"
        self.root_dir = root_dir
        self.K = np.array([[585, 0, 320],
                           [0, 585, 240],
                           [0, 0, 1]], dtype=np.float32)
        interval = 50
        step = 10

        if scene == 'all':
            scenes = [s for s in sorted(train_splits.keys()) if os.path.isdir(os.path.join(root_dir, s))]
        else:
            scenes = [scene] if os.path.isdir(os.path.join(root_dir, scene)) else []

        self.img_pairs, self.depth_pairs, self.T_1_2, self.R_1_2 = [], [], [], []
        for s in scenes:
            sequences = train_splits[s] if split == 'train' else test_splits[s]
            for seq in sequences:
                seq_path = os.path.join(root_dir, s, seq)
                if not os.path.isdir(seq_path):
                    continue
                img_paths = [
                    os.path.join(seq_path, p)
                    for p in os.listdir(seq_path) if p[-9:] == 'color.png']
                img_paths.sort()
                depth_paths = [
                    os.path.join(seq_path, p)
                    for p in os.listdir(seq_path) if p[-9:] == 'depth.png']
                depth_paths.sort()
                pose_paths = [
                    os.path.join(seq_path, p)
                    for p in os.listdir(seq_path) if p[-8:] == 'pose.txt']
                pose_paths.sort()
                num_img = min(len(img_paths), len(pose_paths))
                for i in range(0, num_img - interval, step):
                    self.img_pairs.append((img_paths[i],
                                           img_paths[i + interval]))
                    if len(depth_paths) == len(img_paths):
                        self.depth_pairs.append((depth_paths[i], depth_paths[i + interval]))
                    else:
                        self.depth_pairs.append((None, None))
                    Rt1 = np.loadtxt(pose_paths[i])
                    Rt2 = np.loadtxt(pose_paths[i + interval])
                    Rt_1_2 = np.linalg.inv(Rt2).dot(Rt1)
                    self.R_1_2.append(Rt_1_2[:3, :3])
                    self.T_1_2.append(Rt_1_2[:3, 3])

        print(f"Initialized 7Scenes dataset with {len(self.img_pairs)} image pairs.")

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
            'depth_path1': self.depth_pairs[item][0],
            'depth_path2': self.depth_pairs[item][1],
            'img1': img1,
            'img2': img2,
            'R_1_2': self.R_1_2[item],
            'T_1_2': self.T_1_2[item],
            'K1': self.K,
            'K2': self.K,
        }

        return outputs

    def __len__(self):
        return len(self.img_pairs)
