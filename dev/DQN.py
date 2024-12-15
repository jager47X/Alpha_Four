import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 7)

    def forward(self, x):
        x = x.view(-1, 6 * 7)  # Flatten the input

        # Check batch size and bypass BatchNorm for single inputs
        x = torch.relu(self.bn1(self.fc1(x)) if x.size(0) > 1 else self.fc1(x))
        x = torch.relu(self.bn2(self.fc2(x)) if x.size(0) > 1 else self.fc2(x))
        x = torch.relu(self.bn3(self.fc3(x)) if x.size(0) > 1 else self.fc3(x))
        x = torch.relu(self.bn4(self.fc4(x)) if x.size(0) > 1 else self.fc4(x))
        x = torch.relu(self.bn5(self.fc5(x)) if x.size(0) > 1 else self.fc5(x))
        return self.fc6(x)
