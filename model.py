import torch
import torch.nn as nn


class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        # The classical layers remain the same
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Input to fc1 is 64 channels * 3x3 feature map size after pooling
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # The input 'x' is now the pre-processed tensor from the quanv_layer
        # Its shape is (batch_size, 14, 14, 4)

        # Reshape for PyTorch's Conv2d: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Pass through the classical layers as before
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)  # Flatten

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
