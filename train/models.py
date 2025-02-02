import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block that applies two 3×3 convolutions with batch normalization 
    and LeakyReLU activations. If the number of input channels differs from the output,
    a projection via a 1×1 convolution is applied to the identity.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If the channel dimensions differ, project the identity to match.
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        if self.projection is not None:
            identity = self.projection(identity)
        out += identity
        out = F.leaky_relu(out, negative_slope=0.01)
        return out

class DQN(nn.Module):
    """
    A Dueling DQN for Connect 4 that combines convolutional layers,
    residual blocks, and separate advantage/value streams.
    """
    def __init__(self, input_shape=(6, 7), num_actions=7, dropout_prob=0.2, device=None):
        super(DQN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.num_actions = num_actions

        # -- Convolutional Feature Extraction --
        # Initial convolution layer: from 1 channel to 32 channels.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks: first block increases channels from 32 to 64,
        # second block processes the features at 64 channels.
        self.resblock1 = ResidualBlock(32, 64)
        self.resblock2 = ResidualBlock(64, 64)
        
        # -- Dueling Network Head --
        # After the conv/residual layers, the feature map size is (64, 6, 7).
        # Flattened size: 64 * 6 * 7.
        flattened_size = 64 * input_shape[0] * input_shape[1]
        
        # Advantage Stream:
        self.fc_adv1 = nn.Linear(flattened_size, 128)
        self.bn_adv1 = nn.BatchNorm1d(128)
        self.fc_adv2 = nn.Linear(128, num_actions)
        
        # Value Stream:
        self.fc_val1 = nn.Linear(flattened_size, 128)
        self.bn_val1 = nn.BatchNorm1d(128)
        self.fc_val2 = nn.Linear(128, 1)
        
        # Dropout for regularization in the fully connected layers.
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # Ensure the input is in shape [batch, 1, rows, columns].
        # Handle cases for single board (2D tensor) or a batch (3D tensor).
        if len(x.shape) == 2:         # [rows, columns]
            x = x.unsqueeze(0).unsqueeze(0)  # becomes [1, 1, rows, columns]
        elif len(x.shape) == 3:       # [batch, rows, columns]
            x = x.unsqueeze(1)       # becomes [batch, 1, rows, columns]

        # Pass through convolutional layer with LeakyReLU.
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)  # [batch, 32, rows, columns]
        # Apply residual blocks.
        x = self.resblock1(x)  # [batch, 64, rows, columns]
        x = self.resblock2(x)  # [batch, 64, rows, columns]
        
        # Flatten the feature maps.
        x = x.view(x.size(0), -1)  # [batch, flattened_size]
        
        # -- Advantage Branch --
        adv = F.leaky_relu(self.bn_adv1(self.fc_adv1(x)), negative_slope=0.01)
        adv = self.dropout(adv)
        adv = self.fc_adv2(adv)
        
        # -- Value Branch --
        val = F.leaky_relu(self.bn_val1(self.fc_val1(x)), negative_slope=0.01)
        val = self.dropout(val)
        val = self.fc_val2(val)
        
        # Combine streams to compute final Q-values:
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q
