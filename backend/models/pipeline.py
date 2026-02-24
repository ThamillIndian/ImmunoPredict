import yaml
import torch
import numpy as np
import joblib
import os
import yaml
from backend.models.encoder import ImmuneEncoder
from backend.ode.ode_system import simulate_trajectory, simulate_at_timepoints
from backend.models.decision import get_risk_tier
from backend.ode.monte_carlo import monte_carlo_trajectories

class ImmunoPredictPipeline:
    """
    The main Hybrid Model Pipeline.
    Process: Raw Early Data -> Scaler -> Encoder -> Theta -> Un-scale -> ODE Simulator -> Risk Decision
    """
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cpu")
        
        # Load Scaler
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
        self.scaler = joblib.load(scaler_path)
        
        # Determine input dimension from scaler
        input_dim = self.scaler.n_features_in_
        
        self.encoder = ImmuneEncoder(
            input_dim=input_dim,
            hidden_dims=self.config['training']['encoder']['hidden_layers']
        )
        self.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.encoder.eval()
        
        # Population Means for un-scaling
        self.theta_means = np.array([
            self.config['theta']['activation']['population_mean'],
            self.config['theta']['prod']['population_mean'],
            self.config['theta']['decay']['population_mean']
        ])
        
    def predict_patient(self, patient_features_df, vaccine_type):
        """
        Runs one patient through the full pipeline.
        patient_features_df: Raw wide features (1 row)
        """
        # 1. Scaling: Raw -> Standardized
        x_scaled = self.scaler.transform(patient_features_df)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        
        # 2. AI Encoder: Predict Normalized Theta
        with torch.no_grad():
            theta_norm = self.encoder(x_tensor).numpy()[0]
            
        # 3. Un-scale: Normalized -> Biological Scale
        theta_pred = theta_norm * self.theta_means
        
        # We assume 10% uncertainty around the un-scaled mean
        means = theta_pred
        stds = theta_pred * 0.1
        
        # 4. ODE Simulator: Predict Trajectories & Uncertainty
        stats = monte_carlo_trajectories(
            means, stds, vaccine_type, self.config['timepoints'], self.config
        )
        
        # 5. Decision Support: Get Risk Tier
        # We look at Day 28 (index 5 in our timepoints [0,1,3,7,14,28,90])
        idx_28 = 5 
        titer_28 = stats['A']['median'][idx_28]
        ci_28 = (stats['A']['p05'][idx_28], stats['A']['p95'][idx_28])
        
        risk = get_risk_tier(titer_28, ci_28, self.config)
        
        return {
            'predicted_theta': {
                'activation': float(means[0]),
                'prod': float(means[1]),
                'decay': float(means[2])
            },
            'predicted_titer_28': float(titer_28),
            'confidence_interval_28': [float(ci_28[0]), float(ci_28[1])],
            'risk_assessment': risk,
            'full_trajectory': {
                'days': self.config['timepoints'],
                'median': stats['A']['median'].tolist(),
                'p5': stats['A']['p05'].tolist(),
                'p95': stats['A']['p95'].tolist()
            }
        }
