import pandas as pd
import torch
from torch.utils.data import Dataset

# Custom dataset class for handling position data from a CSV file
class PositionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.column_names = ['p3x','p3y','p2x','p2y','p1x','p1y','cx','cy']

    # Return the total number of samples in the dataset    
    def __len__(self):
        return len(self.data)
    
    # Retrieve one sample from the dataset at the specified index
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Build a list of image-like tensors for the previous three positions
        input_images = []
        for prefix in ['p3', 'p2', 'p1']:
            x = row[f'{prefix}x']
            y = row[f'{prefix}y']
            # Create a 10x10 tensor filled with -1.0 (default state)
            img = torch.full((10, 10), -1.0, dtype=torch.float32)
            # Set the position to 1 if valid coordinates exist
            if x >= 0 and y >= 0:
                img[y, x] = 1.0
            input_images.append(img)
            
        # The target is created as a single index from the current position (flattened 10x10 grid)
        target = row['cy']*10+row['cx']
        
        # Return the stacked input images and the target as a tensor
        return torch.stack(input_images), torch.tensor(target, dtype=torch.long)