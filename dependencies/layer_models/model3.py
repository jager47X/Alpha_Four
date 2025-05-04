import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) for board games like Connect 4.

    This model takes a single-channel 6x7 board input and outputs Q-values
    for 7 possible actions (columns to drop a piece).

    Architecture Summary:
    - 5 convolutional layers with InstanceNorm and ReLU
    - 1 MaxPool2d layer to downsample spatial dimensions
    - 3-layer fully connected block with GroupNorm and Dropout
    - Final output: Q-values for each possible action
    """

    def __init__(self, in_channels=1, num_actions=7):
        """
        Initialize the DQN model.

        Args:
            in_channels (int): Number of input channels (default: 1 for grayscale board).
            num_actions (int): Number of possible actions (default: 7 for Connect 4 columns).
        """
        super(DQN, self).__init__()

        self.conv_block = nn.Sequential(
            # Conv Layer 1: Initial feature extraction
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),    # Output: (64, 6, 7)
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),             # Output: (128, 6, 7)
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            # Downsampling with MaxPooling (preserves 4 columns with ceil_mode=True)
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),   # Output: (128, 3, 4)

            # Conv Layer 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),            # Output: (128, 3, 4)
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            # Conv Layer 4: Channel expansion
            nn.Conv2d(128, 256, kernel_size=3, padding=1),            # Output: (256, 3, 4)
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),

            # Conv Layer 5: Optional deeper capacity
            nn.Conv2d(256, 256, kernel_size=3, padding=1),            # Output: (256, 3, 4)
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )

        self.fc_block = nn.Sequential(
            # Fully Connected Layer 1
            nn.Dropout(0.4),
            nn.Linear(256 * 3 * 4, 512),  # Flattened size: 3072
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),

            # Fully Connected Layer 2
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            # Fully Connected Layer 3
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),

            # Output Layer: Q-values for each action
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 6, 7)

        Returns:
            torch.Tensor: Q-values for each action (shape: batch_size x num_actions)
        """
        x = self.conv_block(x)                  # Apply convolutional layers
        x = x.view(x.size(0), -1)               # Flatten the output
        return self.fc_block(x)                 # Apply fully connected layers
