import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class Stage2Dataset(Dataset):
    """
    Dataset that pairs wide-format biomarkers (X) with fitted theta (y).
    """
    def __init__(self, X_df, theta_df):
        # Align indices
        common_idx = X_df.index.intersection(theta_df['subject_id'])
        X_df = X_df.loc[common_idx]
        theta_df = theta_df.set_index('subject_id').loc[common_idx]
        
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        
        # We train on the 'fitted' theta from Stage 1
        self.y = torch.tensor(theta_df[['fit_activation', 'fit_prod', 'fit_decay']].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X, theta_df, batch_size=32, train_split=0.8):
    full_dataset = Stage2Dataset(X, theta_df)
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader
