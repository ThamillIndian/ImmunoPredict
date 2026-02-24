import pandas as pd
import os
import json
from backend.models.pipeline import ImmunoPredictPipeline
from backend.train.baseline import prepare_baseline_features
import yaml

def main():
    config_path = 'backend/config.yaml'
    model_path = 'backend/artifacts/stage2/encoder_best.pth'
    data_path = 'backend/data/dataset_train.csv'
    
    if not os.path.exists(model_path):
        print("Error: Encoder model not found. Run Phase 5 training first.")
        return

    # 1. Initialize Pipeline
    print("Initializng Hybrid Pipeline...")
    pipeline = ImmunoPredictPipeline(config_path, model_path)
    
    # 2. Load some test data (first 5 patients)
    df_raw = pd.read_csv(data_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    X, _ = prepare_baseline_features(df_raw, config)
    
    print("\nRunning test predictions for first 3 patients...")
    for i in range(3):
        subject_id = X.index[i]
        vaccine_type = df_raw[df_raw['subject_id'] == subject_id]['vaccine_type'].iloc[0]
        
        # Predict
        result = pipeline.predict_patient(X.iloc[[i]], vaccine_type)
        
        print(f"\n--- Subject: {subject_id} ---")
        print(f"Vaccine: {vaccine_type}")
        print(f"Predicted Theta: {result['predicted_theta']}")
        print(f"Predicted Titer (Day 28): {result['predicted_titer_28']:.2f}")
        print(f"Risk Tier: {result['risk_assessment']['tier']}")
        print(f"Action: {result['risk_assessment']['action']}")
        print(f"Message: {result['risk_assessment']['message']}")

if __name__ == "__main__":
    main()
