"""
Image dataset for the CNN
contain the path of the image, the square as a label, and the points in exact coordinates
"""
import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from skimage import io
from PIL import Image


class RSSIImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

        image = io.imread(img_name)
        image = Image.fromarray(image).convert('RGB')

        label = int(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        px = float(self.annotations.iloc[index, 2])
        py = float(self.annotations.iloc[index, 3])

        point = {'x': px, 'y': py}

        return image, label, point
