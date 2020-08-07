import torch.nn as nn
import torch.nn.functional as F

class DisNet(nn.Module):
    def __init__(self):
        super(DisNet, self).__init__()
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x