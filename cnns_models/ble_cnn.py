"""
The model of the cnn taken by the paper
Hybrid Wireless Fingerprint Indoor Localization Method Based on a Convolutional Neural Network
"""
import torch.nn as nn
import torch.nn.functional as F


class BLEcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 4608)
        self.fc2 = nn.Linear(4608, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc2(self.dropout(self.fc1(x)))
        return x
