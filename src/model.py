# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseCNN is a convolutional neural network with dense concatenation layers
class DenseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First convolution layer: takes 3 input channels, outputs 4 channels
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        # Second convolution layer: takes the concatenated tensor (3+4 channels = 7) to output 10 channels
        self.conv2 = nn.Conv2d(7, 10, kernel_size=3, padding=1)
        # Third convolution layer: takes concatenated output (7+10 = 17 channels) and outputs 4 channels
        self.conv3 = nn.Conv2d(17, 4, kernel_size=3, padding=1)
        # Fourth convolution layer: takes concatenated output (17+4 = 21 channels) and outputs 4 channels
        self.conv4 = nn.Conv2d(21, 4, kernel_size=3, padding=1)
        
        # Flatten layer to reshape the tensor into a vector for fully-connected layers
        self.flatten = nn.Flatten()
        # First linear layer: input features from flattened conv output to 250 neurons
        self.linear1 = nn.Linear(25*10*10, 250)
        # Second linear layer: reduces 250 neurons to 100 output features
        self.linear2 = nn.Linear(250, 100)
        # Dropout layer with 50% probability to help prevent overfitting
        self.dropout = nn.Dropout(0.5)
                  
    # Define the forward pass of the network
    def forward(self, x):
        # Apply first convolution followed by ReLU activation
        out1 = F.relu(self.conv1(x))
        # Concatenate the original input and the first convolution's output along the channel dimension
        cat1 = torch.cat([x, out1], 1)
        
        # Second convolution block with activation and concatenation
        out2 = F.relu(self.conv2(cat1))
        cat2 = torch.cat([cat1, out2], 1)
        
        # Third convolution block with activation and concatenation
        out3 = F.relu(self.conv3(cat2))
        cat3 = torch.cat([cat2, out3], 1)
        
        # Fourth convolution block with activation and concatenation
        out4 = F.relu(self.conv4(cat3))
        cat4 = torch.cat([cat3, out4], 1)
        
        # Flatten the concatenated tensor into a vector
        x = self.flatten(cat4)
        # Apply the first linear (fully-connected) layer and ReLU activation
        x = F.relu(self.linear1(x))
        # Apply dropout for regularization
        x = self.dropout(x)
        # Return the final output after the second linear layer (no activation applied here)
        return self.linear2(x)