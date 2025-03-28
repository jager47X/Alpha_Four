import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=7):
        super(DQN, self).__init__()
        # Input shape: (batch_size, 1, 6, 7)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), # padding to maintain spatial dims
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),  # Spatial dims shrink here
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Output from conv layers: (128, 3, 4) => 128*3*4 = 1536
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 3 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x