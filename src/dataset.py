import pandas as pd
import torch
from torch.utils.data import Dataset

class PositionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.column_names = ['p3x','p3y','p2x','p2y','p1x','p1y','cx','cy']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input tensor
        input_images = []
        for prefix in ['p3', 'p2', 'p1']:
            x = row[f'{prefix}x']
            y = row[f'{prefix}y']
            img = torch.full((10, 10), -1.0, dtype=torch.float32)
            if x >= 0 and y >= 0:
                img[y, x] = 1.0
            input_images.append(img)
            
        # Create target tensor
        target = row['cy']*10+row['cx']
        
        return torch.stack(input_images), torch.tensor(target, dtype=torch.long)