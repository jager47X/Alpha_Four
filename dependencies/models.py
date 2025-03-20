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
        # Projection for channel mismatch
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
        return F.relu(out)

class DQN(nn.Module):
    """
    A deeper CNN-based DQN for Connect 4 that uses:
      - Multiple convolutional layers for spatial feature extraction.
      - A series of residual blocks for deeper representation learning.
      - Additional convolution stages to further refine features.
      - A deeper common fully connected representation.
      - A dueling head to separate the estimation of state value and action advantage.
    """
    def __init__(self, input_shape=(6, 7), num_actions=7, dropout_prob=0.3, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_actions = num_actions

        # -- Convolutional Backbone --
        # Initial convolution: from 1 channel to 32 channels.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Stack of residual blocks to deepen the feature extractor.
        # First set: gradually increasing the feature dimension.
        self.resblocks_stage1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64)
        )
        
        # Additional convolutional stage.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Second set of residual blocks at the higher channel count.
        self.resblocks_stage2 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        
        # Optional: One more convolutional layer to mix features.
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # -- Fully Connected Common Representation --
        # After conv layers, feature map size: (256, rows, columns)
        flattened_size = 256 * input_shape[0] * input_shape[1]
        self.fc_common1 = nn.Linear(flattened_size, 1024)
        self.bn_common1 = nn.BatchNorm1d(1024)
        self.fc_common2 = nn.Linear(1024, 512)
        self.bn_common2 = nn.BatchNorm1d(512)
        self.dropout_common = nn.Dropout(p=dropout_prob)
        
        # -- Advantage Branch --
        self.fc_adv1 = nn.Linear(512, 256)
        self.bn_adv1 = nn.BatchNorm1d(256)
        self.dropout_adv = nn.Dropout(p=dropout_prob)
        self.fc_adv2 = nn.Linear(256, num_actions)
        
        # -- Value Branch --
        self.fc_val1 = nn.Linear(512, 256)
        self.bn_val1 = nn.BatchNorm1d(256)
        self.dropout_val = nn.Dropout(p=dropout_prob)
        self.fc_val2 = nn.Linear(256, 1)

    def forward(self, x):
        # Ensure input shape [batch, 1, rows, columns]
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # --- Convolutional Feature Extraction ---
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, rows, cols)
        x = self.resblocks_stage1(x)         # Increase channels to 64
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, rows, cols)
        x = self.resblocks_stage2(x)         # Remains at 128 channels
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 256, rows, cols)
        
        # Flatten convolutional features.
        x = x.view(x.size(0), -1)
        
        # --- Deep Fully Connected Common Representation ---
        common = F.relu(self.bn_common1(self.fc_common1(x)))
        common = F.relu(self.bn_common2(self.fc_common2(common)))
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
