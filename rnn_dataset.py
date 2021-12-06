"""
Dataset for the rnn
It will have the path of the numpy file corresponding to the matrix, and the value of the point
"""
import os
import pandas as pd
import torch

import numpy as np

from torch.utils.data import Dataset


class RnnDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        matrix_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

        with open(matrix_name, 'rb') as f:
            RSSI = np.load(f)

        px = float(self.annotations.iloc[index, 1])
        py = float(self.annotations.iloc[index, 2])
        point = torch.tensor(np.array([px, py]))

        if self.transform:
            RSSI = self.transform(RSSI)

        RSSI = RSSI.view(10, 5)

        if self.target_transform:
            point = self.target_transform(point)

        return RSSI, point
