import torch
import torch.nn as nn
import torch.nn.functional as F


class WiFicnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool_no_change = nn.MaxPool2d(2, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 18)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.pad(x, (1, 0, 1, 0))  # [left, right, top, bot]
        x = self.pool_no_change(F.relu(self.conv3(x)))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(F.relu(self.fc1(self.flatten(x))))
        x = self.fc2(x)
        return x
