import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1) Convolutional Layers
        # Input shape: (batch_size, 1, 6, 7)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output: (16, 6, 7)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: (32, 6, 7)
        self.bn2 = nn.BatchNorm2d(32)

        # 2) Flattening Layer and Fully Connected Layers
        # After conv2, output shape is (batch_size, 32, 6, 7) => Flatten to (batch_size, 32 * 6 * 7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 6 * 7, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 7)  # 7 possible columns (actions)

        # Dropouts for regularization
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        # Ensure input has the correct shape: [batch_size, 1, 6, 7]
        if len(x.shape) == 5:  # Case: [batch_size, 1, 1, 6, 7]
            x = x.squeeze(2)  # Remove the extra singleton dimension
        if len(x.shape) == 3:  # Case: [batch_size, 6, 7]
            x = x.unsqueeze(1)  # Add channel dimension to make [batch_size, 1, 6, 7]

        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))  # Output: [batch_size, 16, 6, 7]
        x = F.relu(self.bn2(self.conv2(x)))  # Output: [batch_size, 32, 6, 7]

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)            # Flatten to [batch_size, 32 * 6 * 7]
        x = F.relu(self.bn3(self.fc1(x)))    # Fully connected layer 1
        x = F.relu(self.bn4(self.fc2(x)))    # Fully connected layer 2

        # Output layer: Q-values for each action
        x = self.fc3(x)                      # Output: [batch_size, num_actions]
        return x

