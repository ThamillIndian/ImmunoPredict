import pandas as pd
import os
import yaml
import time
from backend.train.stage1_fit_theta import run_parameter_fitting

def load_config():
    with open('backend/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_path = 'backend/data/dataset_train.csv'
    output_path = 'backend/artifacts/stage1/fitted_theta.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Fit Parameters
    start_time = time.time()
    df_fitted = run_parameter_fitting(df, config)
    duration = time.time() - start_time
    
    # 3. Merge with ground truth (for validation)
    # In a real scenario, we wouldn't have 'true_theta' column, 
    # but here we can check how well we recovered the values.
    # Note: dataset_train.csv already contains 'theta_activation', etc.
    df_gt = df[['subject_id', 'theta_activation', 'theta_prod', 'theta_decay']].drop_duplicates()
    df_eval = df_fitted.merge(df_gt, on='subject_id')
    
    # Calculate recovery error
    df_eval['error_activation'] = (df_eval['fit_activation'] - df_eval['theta_activation']).abs()
    df_eval['error_prod'] = (df_eval['fit_prod'] - df_eval['theta_prod']).abs()
    
    # 4. Save
    df_eval.to_csv(output_path, index=False)
    
    print(f"\nStage 1 Fitting Complete in {duration:.1f}s")
    print(f"Fitted parameters saved to: {output_path}")
    print(f"Mean Recovery Error (Activation): {df_eval['error_activation'].mean():.4f}")
    print(f"Mean Recovery Error (Production): {df_eval['error_prod'].mean():.4f}")

if __name__ == "__main__":
    main()
