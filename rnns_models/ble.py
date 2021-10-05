import torch
import torch.nn as nn
import torch.nn.functional as F


class BLErnn(nn.Module):
    def __init__(self, linear_mul=4, lstm_size=32):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(5, (linear_mul*5))
        self.lstm = torch.nn.LSTM((linear_mul*5), lstm_size, batch_first=True)
        self.fc2 = nn.Linear(lstm_size, 2)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x, _ = self.lstm(x)
        x = self.fc2(x[:, -1, :])
        return x
