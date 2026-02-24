import pandas as pd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from backend.models.pipeline import ImmunoPredictPipeline
from backend.train.baseline import prepare_baseline_features
from sklearn.metrics import roc_curve, auc, confusion_matrix

def main():
    config_path = 'backend/config.yaml'
    model_path = 'backend/artifacts/stage2/encoder_best.pth'
    data_path = 'backend/data/dataset_train.csv'
    output_dir = 'backend/artifacts/evaluation/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = ImmunoPredictPipeline(config_path, model_path)
    df_raw = pd.read_csv(data_path)
    X, _ = prepare_baseline_features(df_raw, config)
    
    # Ground Truth
    df_gt = df_raw[df_raw['day'] == 28][['subject_id', 'antibody_titer', 'low_responder_label', 'vaccine_type', 'age']].set_index('subject_id')
    common_idx = X.index.intersection(df_gt.index)
    X = X.loc[common_idx]
    df_gt = df_gt.loc[common_idx]
    
    results = []
    print(f"Running comprehensive evaluation on {len(common_idx)} subjects...")
    
    for subject_id in common_idx:
        v_type = df_gt.loc[subject_id, 'vaccine_type']
        pred = pipeline.predict_patient(X.loc[[subject_id]], v_type)
        
        results.append({
            'subject_id': subject_id,
            'vaccine': v_type,
            'age': df_gt.loc[subject_id, 'age'],
            'true_titer': df_gt.loc[subject_id, 'antibody_titer'],
            'pred_titer': pred['predicted_titer_28'],
            'true_label': df_gt.loc[subject_id, 'low_responder_label'],
            'pred_tier': pred['risk_assessment']['tier'],
            'pred_prob': 1.0 if pred['risk_assessment']['tier'] == 'HIGH' else (0.5 if pred['risk_assessment']['tier'] == 'MEDIUM' else 0.0)
        })
    
    res_df = pd.DataFrame(results)
    
    # --- FIGURE 1: Overall ROC Curve ---
    fpr, tpr, _ = roc_curve(res_df['true_label'], res_df['pred_prob'])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Hybrid Model: ROC Curve (Day 28 Prediction)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '01_overall_roc.png'))
    
    # --- FIGURE 2: Titer Parity by Vaccine ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=res_df, x='true_titer', y='pred_titer', hue='vaccine', style='vaccine', alpha=0.7)
    plt.plot([0, 150], [0, 150], 'r--', alpha=0.5)
    plt.title('Titer Prediction Accuracy by Vaccine Type')
    plt.xlabel('Actual Titer')
    plt.ylabel('Predicted Titer')
    plt.savefig(os.path.join(output_dir, '02_parity_by_vaccine.png'))
    
    # --- FIGURE 3: Error vs Age ---
    res_df['error'] = res_df['pred_titer'] - res_df['true_titer']
    plt.figure(figsize=(10, 6))
    sns.regplot(data=res_df, x='age', y='error', scatter_kws={'alpha':0.3})
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Prediction Error vs. Subject Age')
    plt.savefig(os.path.join(output_dir, '03_error_vs_age.png'))

    # --- FIGURE 4: Risk Tier Confusion Matrix ---
    # Convert tiers to Numeric for simple CM
    tier_map = {'LOW': 0, 'MEDIUM': 0.5, 'HIGH': 1}
    # For CM, let's just look at High vs Not
    y_true = res_df['true_label']
    y_pred_bin = (res_df['pred_tier'] == 'HIGH').astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted HIGH RISK')
    plt.ylabel('Actual LOW RESPONDER')
    plt.title('Clinical Risk Classification matrix')
    plt.savefig(os.path.join(output_dir, '04_confusion_matrix.png'))

    print(f"Comprehensive figures saved to {output_dir}")

if __name__ == "__main__":
    main()
