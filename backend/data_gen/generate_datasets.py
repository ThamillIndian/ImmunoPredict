import pandas as pd
import numpy as np
import os
from .config_loader import load_config
from .population import generate_demographics, generate_theta
from .biomarkers import generate_biomarkers, compute_derived_scores, apply_missingness, apply_noise
from backend.ode.ode_system import simulate_at_timepoints

def generate_cohort(cohort_name, vaccine_type, config):
    """
    Generates a full dataset for a single cohort.
    """
    cohort_cfg = config['population'][cohort_name]
    n_subjects = cohort_cfg['n_subjects']
    timepoints = config['timepoints']
    
    # 1. Generate Population
    print(f"  Generating population for {cohort_name}...")
    demographics = generate_demographics(cohort_cfg, n_subjects)
    population = generate_theta(demographics, config['theta'], config['theta_modifiers'])
    
    # 2. Simulate for each patient
    print(f"  Simulating immune responses...")
    rows = []
    for _, patient in population.iterrows():
        theta_dict = {
            'activation': patient['theta_activation'],
            'prod': patient['theta_prod'],
            'decay': patient['theta_decay']
        }
        
        # ODE Simulation
        sol = simulate_at_timepoints(theta_dict, vaccine_type, timepoints, config)
        
        # Generate Biomarkers per timepoint
        for i, t in enumerate(timepoints):
            states = {'I': sol['I'][i], 'P': sol['P'][i], 'A': sol['A'][i]}
            bms = generate_biomarkers(states, cohort_cfg['noise_level'], config)
            
            row = {
                'subject_id': int(patient['subject_id']),
                'cohort': cohort_name,
                'vaccine_type': vaccine_type,
                'day': int(t),
                'age': int(patient['age']),
                'sex': int(patient['sex']),
                'bmi': float(patient['bmi']),
                'comorbidity_score': int(patient['comorbidity_score']),
                'theta_activation': float(patient['theta_activation']),
                'theta_prod': float(patient['theta_prod']),
                'theta_decay': float(patient['theta_decay']),
                **bms
            }
            
            # Antibody Titer
            if t in config['titer_days']:
                # Add noise to titer too
                row['antibody_titer'] = apply_noise(states['A'], cohort_cfg['noise_level'])
            else:
                row['antibody_titer'] = np.nan
                
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # 3. Compute Derived Scores
    i_score, a_score = compute_derived_scores(df)
    df['innate_score'] = i_score
    df['adaptive_score'] = a_score
    
    # 4. Compute Labels (based on Day 28 titer)
    # Get A28 per subject
    a28_map = df[df['day'] == 28].set_index('subject_id')['antibody_titer']
    threshold = config['decision']['low_responder_threshold']
    df['low_responder_label'] = df['subject_id'].map(lambda sid: 1 if a28_map[sid] < threshold else 0)
    
    # 5. Apply Missingness
    df = apply_missingness(df, cohort_cfg['missingness_rate'])
    
    return df

def main():
    config = load_config()
    data_dir = 'backend/data/'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting Synthetic Data Generation...")
    
    # Cohort 1: Train (Vaccine A)
    df_train = generate_cohort('train', 'A', config)
    df_train.to_csv(f"{data_dir}dataset_train.csv", index=False)
    print(f"[OK] Saved {data_dir}dataset_train.csv")
    
    # Cohort 2: Test Shift (Vaccine A, older/sicker)
    df_shift = generate_cohort('test_shift', 'A', config)
    df_shift.to_csv(f"{data_dir}dataset_test_shift.csv", index=False)
    print(f"[OK] Saved {data_dir}dataset_test_shift.csv")
    
    # Cohort 3: New Vaccine (Vaccine B)
    df_new = generate_cohort('new_vaccine', 'B', config)
    df_new.to_csv(f"{data_dir}dataset_new_vaccine.csv", index=False)
    print(f"[OK] Saved {data_dir}dataset_new_vaccine.csv")
    
    print("\nData Generation Complete.")

if __name__ == "__main__":
    main()
