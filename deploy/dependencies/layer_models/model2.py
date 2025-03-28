import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=7):
        super(DQN, self).__init__()
        self.conv_block = nn.Sequential(
            # First conv block: maintains the input dimensions (6x7)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),   # Output: (64, 6, 7)
            # Replace BatchNorm2d with InstanceNorm2d for small batch stability
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
                
            # Second conv block: still preserves dims, then pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),             # Output: (128, 6, 7)
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                       # Output: (128, 3, 3)
            
            # Third conv block: further feature extraction on reduced dims
            nn.Conv2d(128, 128, kernel_size=3, padding=1),            # Output: (128, 3, 3)
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            
            # Fourth conv block: increases feature channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),            # Output: (256, 3, 3)
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.fc_block = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 3 * 3, 512),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.fc_block(x)
        return x
