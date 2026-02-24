import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib
import os
import json

def prepare_baseline_features(df, config):
    """
    Pivots long-format data (multiple rows per patient) to wide-format (one row per patient)
    using only early-day biomarkers (Day 0, 1, 3, 7).
    """
    early_days = config['early_days']
    
    # 1. Filter for early days and necessary columns
    cols = ['subject_id', 'day', 'age', 'sex', 'bmi', 'comorbidity_score', 
            'cytokine_il6', 'cytokine_tnfa', 'cytokine_ifng', 
            'wbc', 'lymphocytes', 'neutrophils']
    
    df_early = df[df['day'].isin(early_days)][cols].copy()
    
    # 2. Pivot biomarkers
    biomarker_cols = ['cytokine_il6', 'cytokine_tnfa', 'cytokine_ifng', 'wbc', 'lymphocytes', 'neutrophils']
    df_pivot = df_early.pivot(index='subject_id', columns='day', values=biomarker_cols)
    
    # Flatten multi-index columns: 'cytokine_il6_0', 'cytokine_il6_1', etc.
    df_pivot.columns = [f"{col}_{day}" for col, day in df_pivot.columns]
    
    # 3. Join back demographics (from Day 0)
    df_dem = df_early[df_early['day'] == 0][['subject_id', 'age', 'sex', 'bmi', 'comorbidity_score']].set_index('subject_id')
    X = df_dem.join(df_pivot)
    
    # 4. Handle Missingness: Simple imputation (mean) and mask columns
    for col in X.columns:
        if X[col].isnull().any():
            X[f"{col}_observed"] = X[col].notnull().astype(int)
            X[col] = X[col].fillna(X[col].mean())
            
    # 5. Get Target (Day 28 titer)
    y = df[df['day'] == 28].set_index('subject_id')[['antibody_titer', 'low_responder_label']]
    
    # Ensure indices align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    return X, y

def train_and_evaluate(X, y_titer, y_label, config, output_dir):
    """
    Trains XGBoost and MLP baselines using 5-fold CV.
    """
    kf = KFold(n_splits=config['training']['baseline'].get('cv_folds', 5), shuffle=True, random_state=42)
    
    results = {
        'xgboost': {'mae': [], 'rmse': [], 'r2': [], 'auc': [], 'f1': []},
        'mlp': {'mae': [], 'rmse': [], 'r2': [], 'auc': [], 'f1': []}
    }
    
    scaler = StandardScaler()
    threshold = config['decision']['low_responder_threshold']
    
    print(f"Starting Baseline Training ({kf.n_splits}-fold CV)...")
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_titer.iloc[train_idx], y_titer.iloc[val_idx]
        l_train, l_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
        
        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # --- XGBoost ---
        xgb_model = xgb.XGBRegressor(**config['training']['baseline']['xgboost'])
        xgb_model.fit(X_train_scaled, y_train)
        preds = xgb_model.predict(X_val_scaled)
        
        results['xgboost']['mae'].append(mean_absolute_error(y_val, preds))
        results['xgboost']['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
        results['xgboost']['r2'].append(r2_score(y_val, preds))
        
        # Classification from regression
        labels_pred = (preds < threshold).astype(int)
        results['xgboost']['f1'].append(f1_score(l_val, labels_pred))
        try:
            results['xgboost']['auc'].append(roc_auc_score(l_val, -preds)) # inverse titer as prob
        except: results['xgboost']['auc'].append(0.5)

        # --- MLP ---
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train.values.ravel())
        preds = mlp.predict(X_val_scaled)
        
        results['mlp']['mae'].append(mean_absolute_error(y_val, preds))
        results['mlp']['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
        results['mlp']['r2'].append(r2_score(y_val, preds))
        
        labels_pred = (preds < threshold).astype(int)
        results['mlp']['f1'].append(f1_score(l_val, labels_pred))
        try:
            results['mlp']['auc'].append(roc_auc_score(l_val, -preds))
        except: results['mlp']['auc'].append(0.5)

    # Aggregate
    final_metrics = {}
    for model in results:
        final_metrics[model] = {k: float(np.mean(v)) for k, v in results[model].items()}
        print(f"  {model.upper()}: R² = {final_metrics[model]['r2']:.3f}, MAE = {final_metrics[model]['mae']:.3f}, AUC = {final_metrics[model]['auc']:.3f}")
        
    # Final train on all data and save
    X_full_scaled = scaler.fit_transform(X)
    final_xgb = xgb.XGBRegressor(**config['training']['baseline']['xgboost'])
    final_xgb.fit(X_full_scaled, y_titer)
    
    joblib.dump(final_xgb, os.path.join(output_dir, 'xgboost_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'baseline_scaler.pkl'))
    
    with open(os.path.join(output_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    return final_metrics, final_xgb, X.columns.tolist()
