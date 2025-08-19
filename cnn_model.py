import torch.nn as nn
import torch


class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        # The classical layers remain the same
        self.feature_extractor = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Input to fc1 is 64 channels * 3x3 feature map size after pooling
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Classical Conv2d layer that outputs tensor with the same dimensions as the quanv layer with 4 qubits
        x = self.feature_extractor(x)

        # Pass through the same convolution layers as the QNN architecture
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)  # Flatten

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
