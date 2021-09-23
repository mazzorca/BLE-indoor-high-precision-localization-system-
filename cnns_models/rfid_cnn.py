import torch
import torch.nn as nn
import torch.nn.functional as F


class RFIDcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 40, 5)
        self.conv3 = nn.Conv2d(40, 16, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        return x