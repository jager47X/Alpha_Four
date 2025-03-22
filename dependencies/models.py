import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=7):
        super(DQN, self).__init__()
        # Input expected shape: (batch_size, 1, 6, 7)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # After conv layers, flatten => feed into two linear layers.
        # For a 6x7 board, the final conv output is (64, 2, 3) => 64*2*3 = 384
        self.fc1 = nn.Linear(64 * 2 * 3, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        # x: (batch_size, 1, 6, 7)
        x = F.relu(self.conv1(x))          # -> (32, 4, 5)
        x = F.relu(self.conv2(x))          # -> (64, 2, 3)
        x = x.view(x.size(0), -1)          # Flatten -> (batch_size, 384)
        x = F.relu(self.fc1(x))            # -> (batch_size, 64)
        x = self.fc2(x)                    # -> (batch_size, 7) => Q-values
        return x

# Usage:
# model = Connect4CNN()
# output = model(torch.randn(1, 1, 6, 7))  # example forward pass
# print(output.shape)  # should be [1, 7]
