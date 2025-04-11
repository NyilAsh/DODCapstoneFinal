# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(7, 10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(17, 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(21, 4, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25*10*10, 250)  
        self.linear2 = nn.Linear(250, 100)       
        self.dropout = nn.Dropout(0.5) 
                  
    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        cat1 = torch.cat([x, out1], 1)
        
        out2 = F.relu(self.conv2(cat1))
        cat2 = torch.cat([cat1, out2], 1)
        
        out3 = F.relu(self.conv3(cat2))
        cat3 = torch.cat([cat2, out3], 1)
        
        out4 = F.relu(self.conv4(cat3))
        cat4 = torch.cat([cat3, out4], 1)
        
        x = self.flatten(cat4)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)