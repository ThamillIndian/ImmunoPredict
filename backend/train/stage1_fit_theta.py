import numpy as np
import pandas as pd
from scipy.optimize import minimize
from backend.ode.ode_system import simulate_at_timepoints
from backend.data_gen.biomarkers import generate_biomarkers
import os
import concurrent.futures

def loss_function(theta_vals, patient_data, config, vaccine_type):
    """
    Computes MSE between observed biomarkers and ODE-predicted biomarkers.
    theta_vals: [activation, prod, decay]
    """
    # 1. Unpack theta
    theta = {
        'activation': theta_vals[0],
        'prod': theta_vals[1],
        'decay': theta_vals[2]
    }
    
    # 2. Simulate ODE
    # We simulate for the specific days in the patient's record
    days = sorted(patient_data['day'].unique())
    results = simulate_at_timepoints(theta, vaccine_type, days, config)
    
    # 3. Map to Biomarkers (Latent -> Observable)
    # Note: We don't apply noise here since we are fitting to the noisy data
    pred_bms = generate_biomarkers(results, noise_level=0.0, config=config)
    
    # 4. Compute Loss
    # We only fit to columns we have observations for
    biomarker_cols = [
        'cytokine_il6', 'cytokine_tnfa', 'cytokine_ifng', 
        'wbc', 'lymphocytes', 'neutrophils'
    ]
    
    total_loss = 0.0
    valid_counts = 0
    
    for col in biomarker_cols:
        obs = patient_data[col].values
        pred = pred_bms[col]
        
        # Handle missingness (NaNs)
        mask = ~np.isnan(obs)
        if mask.any():
            # Normalized MSE (scale effects differ across biomarkers)
            # Dividing by the mean of observed values to normalize
            ref_val = np.mean(obs[mask]) + 1e-6
            total_loss += np.mean(((obs[mask] - pred[mask]) / ref_val)**2)
            valid_counts += 1
            
    # Also fit to antibody titer (this is often our 'ground truth' anchor)
    titer_obs = patient_data['antibody_titer'].values
    titer_pred = results['A']
    mask_t = ~np.isnan(titer_obs)
    if mask_t.any():
        ref_t = np.mean(titer_obs[mask_t]) + 1e-6
        total_loss += np.mean(((titer_obs[mask_t] - titer_pred[mask_t]) / ref_t)**2)
        valid_counts += 1

    return total_loss / valid_counts if valid_counts > 0 else 1e9

def fit_single_patient(subject_id, df_subset, config):
    """
    Performs L-BFGS-B optimization to find the best theta for one subject.
    """
    vaccine_type = df_subset['vaccine_type'].iloc[0]
    
    # Priors / Initial Guesses from config
    t_cfg = config['theta']
    x0 = [
        t_cfg['activation']['population_mean'],
        t_cfg['prod']['population_mean'],
        t_cfg['decay']['population_mean']
    ]
    
    bounds = [
        t_cfg['activation']['bounds'],
        t_cfg['prod']['bounds'],
        t_cfg['decay']['bounds']
    ]
    
    # Optimize
    res = minimize(
        loss_function, 
        x0, 
        args=(df_subset, config, vaccine_type),
        method=config['training']['stage1']['optimizer'],
        bounds=bounds,
        options={'maxiter': 50}
    )
    
    return {
        'subject_id': subject_id,
        'fit_activation': res.x[0],
        'fit_prod': res.x[1],
        'fit_decay': res.x[2],
        'fit_loss': res.fun,
        'fit_success': res.success
    }

def run_parameter_fitting(df, config, max_workers=None):
    """
    Orchestrates fitting for all subjects using a process pool.
    """
    subjects = df['subject_id'].unique()
    print(f"Starting Stage 1 fitting for {len(subjects)} subjects...")
    
    results = []
    
    # Using concurrent.futures for speed
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_single_patient, sid, df[df['subject_id'] == sid], config): sid 
            for sid in subjects
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            if (i+1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(subjects)} fits complete")
                
    return pd.DataFrame(results)
