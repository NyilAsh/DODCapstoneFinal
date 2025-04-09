import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(11, 20, kernel_size=3, padding=1)
        
        # Modified final layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(31 * 10 * 10, 100)  # 100 positions + 1 no-position class
        # Removed unflatten and final conv layers

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        cat1 = torch.cat([x, out1], 1)
        
        out2 = F.relu(self.conv2(cat1))
        cat2 = torch.cat([cat1, out2], 1)
        
        x = self.flatten(cat2)
        return self.linear1(x)  # Direct classification output