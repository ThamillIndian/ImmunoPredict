import torch
import torch.nn as nn
import torch.nn.functional as F

class ImmuneEncoder(nn.Module):
    """
    Neural Network that maps longitudinal early biomarkers to ODE parameters (theta).
    Input Shape: (Batch, Days * Biomarkers)
    Output Shape: (Batch, 3) -> [activation, prod, decay]
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(ImmuneEncoder, self).__init__()
        
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Final head to predict 3 parameters
        self.head = nn.Linear(curr_dim, 3)
        
    def forward(self, x):
        # x is flattened (batch, features)
        features = self.backbone(x)
        theta_pred = self.head(features)
        
        # We know theta must be positive, so we use Softplus to ensure positivity
        # and match the biological range more naturally.
        return F.softplus(theta_pred)
