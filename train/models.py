import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two 3×3 convolutions, batch normalization,
    and ReLU activations. A 1×1 convolution is used as a projection
    if the input and output channels differ.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        # If channel dimensions differ, use a projection
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.projection is not None:
            identity = self.projection(identity)
        out += identity
        out = F.relu(out)
        return out

class DQN(nn.Module):
    """
    A CNN-based DQN for Connect 4 that uses:
      - Convolutional layers for spatial feature extraction.
      - Residual blocks for deeper representation learning.
      - A dueling head to separate the estimation of state value and action advantage.
    """
    def __init__(self, input_shape=(6, 7), num_actions=7, dropout_prob=0.3, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_actions = num_actions  # Should be 7 for Connect 4

        # -- Convolutional Backbone --
        # Initial convolution: from 1 channel to 32 channels.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=1)  # Output: (32, 6, 7)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Two residual blocks to learn deeper features.
        self.resblock1 = ResidualBlock(32, 64)  # Output: (64, 6, 7)
        self.resblock2 = ResidualBlock(64, 64)  # Output: (64, 6, 7)
        
        # Additional convolution for further processing.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)  # Output: (128, 6, 7)
        self.bn3 = nn.BatchNorm2d(128)
        
        # -- Fully Connected Common Representation --
        # After conv layers, the feature map is (128, 6, 7)
        flattened_size = 128 * input_shape[0] * input_shape[1]  # 128 * 6 * 7
        self.fc_common = nn.Linear(flattened_size, 512)
        self.bn_common = nn.BatchNorm1d(512)
        self.dropout_common = nn.Dropout(p=dropout_prob)
        
        # -- Advantage Branch --
        # This branch outputs a vector of length num_actions (7).
        self.fc_adv1 = nn.Linear(512, 256)
        self.bn_adv1 = nn.BatchNorm1d(256)
        self.dropout_adv = nn.Dropout(p=dropout_prob)
        self.fc_adv2 = nn.Linear(256, num_actions)
        
        # -- Value Branch --
        # This branch outputs a single scalar value.
        self.fc_val1 = nn.Linear(512, 256)
        self.bn_val1 = nn.BatchNorm1d(256)
        self.dropout_val = nn.Dropout(p=dropout_prob)
        self.fc_val2 = nn.Linear(256, 1)

    def forward(self, x):
        # Ensure the input is in the form [batch, 1, rows, columns].
        if len(x.shape) == 2:         # Single board: [rows, columns]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, rows, columns]
        elif len(x.shape) == 3:       # Batch of boards: [batch, rows, columns]
            x = x.unsqueeze(1)  # -> [batch, 1, rows, columns]

        # --- Convolutional Feature Extraction ---
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, 6, 7)
        x = self.resblock1(x)  # (batch, 64, 6, 7)
        x = self.resblock2(x)  # (batch, 64, 6, 7)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 128, 6, 7)
        
        # Flatten the convolutional features.
        x = x.view(x.size(0), -1)  # (batch, flattened_size)
        
        # --- Common Fully Connected Layer ---
        common = F.relu(self.bn_common(self.fc_common(x)))
        common = self.dropout_common(common)
        
        # --- Advantage Branch ---
        adv = F.relu(self.bn_adv1(self.fc_adv1(common)))
        adv = self.dropout_adv(adv)
        adv = self.fc_adv2(adv)
        
        # --- Value Branch ---
        val = F.relu(self.bn_val1(self.fc_val1(common)))
        val = self.dropout_val(val)
        val = self.fc_val2(val)
        
        # --- Combine the Streams ---
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q = val + (adv - adv.mean(dim=1, keepdim=True))
       
        return q