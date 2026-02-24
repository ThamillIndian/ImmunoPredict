import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from backend.train.baseline import prepare_baseline_features, train_and_evaluate

def load_config():
    with open('backend/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_path = 'backend/data/dataset_train.csv'
    output_dir = 'backend/artifacts/baseline/'
    fig_dir = 'backend/artifacts/figures/baseline/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data generation first.")
        return

    # 1. Load Data
    print("Loading training data for baseline...")
    df = pd.read_csv(data_path)
    
    # 2. Prepare Features
    X, y_all = prepare_baseline_features(df, config)
    y_titer = y_all['antibody_titer']
    y_label = y_all['low_responder_label']
    
    print(f"Features prepared. Shape: {X.shape}")
    print(f"Target distribution: {y_label.value_counts().to_dict()}")
    
    # 3. Train
    metrics, model, feature_names = train_and_evaluate(X, y_titer, y_label, config, output_dir)
    
    # 4. Feature Importance Plot (XGBoost)
    plt.figure(figsize=(12, 8))
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(20)
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False)
    plt.title("Top 20 Features — XGBoost Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
    plt.close()
    
    print(f"\nBaseline training complete.")
    print(f"Models and metrics saved to {output_dir}")
    print(f"Importance plot saved to {fig_dir}")

if __name__ == "__main__":
    main()
