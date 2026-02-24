import pandas as pd
import torch
import numpy as np
import os
import yaml
from backend.models.pipeline import ImmunoPredictPipeline
from backend.train.baseline import prepare_baseline_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
import matplotlib.pyplot as plt

def main():
    config_path = 'backend/config.yaml'
    model_path = 'backend/artifacts/stage2/encoder_best.pth'
    data_path = 'backend/data/dataset_train.csv'
    output_dir = 'backend/artifacts/evaluation/'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Init Pipeline
    pipeline = ImmunoPredictPipeline(config_path, model_path)
    
    # 2. Load Data
    df_raw = pd.read_csv(data_path)
    X, _ = prepare_baseline_features(df_raw, config)
    
    # Ground Truth: Day 28 Titer and Labels
    # Extract only one row per subject from the raw data for true labels
    df_gt = df_raw[df_raw['day'] == 28][['subject_id', 'antibody_titer', 'low_responder_label']].set_index('subject_id')
    
    # Align X and GT
    common_idx = X.index.intersection(df_gt.index)
    X = X.loc[common_idx]
    df_gt = df_gt.loc[common_idx]
    
    print(f"Evaluating Hybrid Model on {len(common_idx)} subjects...")
    
    results = []
    
    for i, subject_id in enumerate(common_idx):
        vaccine_type = df_raw[df_raw['subject_id'] == subject_id]['vaccine_type'].iloc[0]
        
        # Prediction
        pred = pipeline.predict_patient(X.loc[[subject_id]], vaccine_type)
        
        results.append({
            'subject_id': subject_id,
            'true_titer': df_gt.loc[subject_id, 'antibody_titer'],
            'pred_titer': pred['predicted_titer_28'],
            'true_label': df_gt.loc[subject_id, 'low_responder_label'],
            'pred_tier': pred['risk_assessment']['tier'],
            'pred_prob': 1.0 if pred['risk_assessment']['tier'] == 'HIGH' else (0.5 if pred['risk_assessment']['tier'] == 'MEDIUM' else 0.0)
        })
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(common_idx)} subjects...")
            
    res_df = pd.DataFrame(results)
    
    # 3. Calculate Metrics
    y_true = res_df['true_titer']
    y_pred = res_df['pred_titer']
    
    metrics = {
        'Hybrid_RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'Hybrid_MAE': float(mean_absolute_error(y_true, y_pred)),
        'Hybrid_R2': float(r2_score(y_true, y_pred)),
        'Hybrid_AUC': float(roc_auc_score(res_df['true_label'], res_df['pred_prob']))
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Standard ML comparison (placeholder - in real life we'd load the XGBoost results)
    # Based on previous run: XGB R2 = 0.26, MAE = 18.3, AUC = 0.79
    
    # 4. Save results
    res_df.to_csv(os.path.join(output_dir, 'hybrid_v_gt.csv'), index=False)
    with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f)
        
    # 5. Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 150], [0, 150], 'r--')
    plt.xlabel("Actual Antibody Titer (Day 28)")
    plt.ylabel("Predicted Antibody Titer (Hybrid)")
    plt.title("Hybrid Model: Accuracy Verification")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'parity_plot.png'))
    
    print(f"\nEvaluation plots saved to {output_dir}")

if __name__ == "__main__":
    main()
