import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from backend.models.encoder import ImmuneEncoder
from backend.models.dataset import Stage2Dataset
from torch.utils.data import DataLoader
from backend.train.baseline import prepare_baseline_features

def load_config():
    with open('backend/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_encoder():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Files
    data_path = 'backend/data/dataset_train.csv'
    theta_path = 'backend/artifacts/stage1/fitted_theta.csv'
    output_dir = 'backend/artifacts/stage2/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading data for Stage 2...")
    df_raw = pd.read_csv(data_path)
    df_theta = pd.read_csv(theta_path)
    
    # 2. Prepare Features (Wide format)
    X, _ = prepare_baseline_features(df_raw, config)
    
    # Align indices
    common_idx = X.index.intersection(df_theta['subject_id'])
    X = X.loc[common_idx]
    df_theta = df_theta.set_index('subject_id').loc[common_idx]
    
    # 3. Scaling
    # Input Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print(f"Saved input scaler to {output_dir}")
    
    # Target Scaling (Scale by population mean to normalize loss importance)
    # theta = [activation, prod, decay]
    means = np.array([
        config['theta']['activation']['population_mean'],
        config['theta']['prod']['population_mean'],
        config['theta']['decay']['population_mean']
    ])
    
    y = df_theta[['fit_activation', 'fit_prod', 'fit_decay']].values
    y_scaled = y / means # Simple division scaling
    
    # 4. Create Dataloaders
    cfg = config['training']['encoder']
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'])
    
    # 5. Initialize Model
    model = ImmuneEncoder(
        input_dim=X.shape[1], 
        hidden_dims=cfg['hidden_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()
    
    # 6. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    early_stop_count = 0
    
    print(f"Starting Scaled Encoder Training on {device}...")
    for epoch in range(cfg['epochs']):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item())
                
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
            
        # Early Stopping & Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'encoder_best.pth'))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= cfg['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title("Scaled Encoder Training Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    print(f"Training Complete. Model saved to {output_dir}")

if __name__ == "__main__":
    train_encoder()

if __name__ == "__main__":
    train_encoder()
