import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1) Convolutional Layers
        # Input shape: (batch_size, 1, 6, 7)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 2) Fully Connected Layers
        # After conv2, shape is (batch_size, 32, 6, 7)
        # Flattened = 32 * 6 * 7 = 1344
        self.fc1 = nn.Linear(32 * 6 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 7)  # 7 possible columns

        # Dropouts (optionally use for regularization)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        x is expected to be shape: (batch_size, 6, 7)
        We'll add a channel dimension to get (batch_size, 1, 6, 7).
        """
        x = x.unsqueeze(1)  # Insert channel dim at position 1

        # Convolution blocks
        x = self.conv1(x)               # shape: (batch_size, 16, 6, 7)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)               # shape: (batch_size, 32, 6, 7)
        x = self.bn2(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)       # shape: (batch_size, 32*6*7)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Final output => Q-values for each of the 7 columns
        x = self.fc3(x)
        return x
