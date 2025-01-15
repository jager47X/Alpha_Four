import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape=(6, 7), num_actions=7, dropout_prob=0.3, device=None):
        """
        A deeper and wider DQN for Connect 4.

        Args:
            input_shape (tuple): Shape of the input (rows, columns).
            num_actions (int): Number of possible actions (e.g., columns in Connect 4).
            dropout_prob (float): Dropout probability for regularization.
            device (torch.device): Device to use for computations (CPU or GPU).
        """
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_actions = num_actions

        # 1) Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 6, 7)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 6, 7)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 6, 7)
        self.bn3 = nn.BatchNorm2d(128)

        # 2) Flattening Layer and Fully Connected Layers
        # After conv3, output shape is (batch_size, 128, 6, 7) => Flatten to (batch_size, 128 * 6 * 7)
        flattened_size = 128 * input_shape[0] * input_shape[1]  # 128 * 6 * 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_actions)  # Output: Q-values for each action

        # Dropouts for regularization
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, rows, columns].

        Returns:
            torch.Tensor: Q-values for each action of shape [batch_size, num_actions].
        """
        # Ensure input has the correct shape: [batch_size, 1, rows, columns]
        if len(x.shape) == 5:  # Case: [batch_size, 1, 1, rows, columns]
            x = x.squeeze(2)
        if len(x.shape) == 3:  # Case: [batch_size, rows, columns]
            x = x.unsqueeze(1)  # Add channel dimension to make [batch_size, 1, rows, columns]

        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 32, rows, columns)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, rows, columns)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 128, rows, columns)

        # Flatten and fully connected layers
        x = self.flatten(x)                  # (batch_size, 128 * rows * columns)
        x = F.relu(self.bn4(self.fc1(x)))    # Fully connected layer 1
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))    # Fully connected layer 2
        x = self.dropout2(x)
        x = F.relu(self.bn6(self.fc3(x)))    # Fully connected layer 3
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)                      # Output: (batch_size, num_actions)
        return x
