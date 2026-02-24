import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

sns.set_theme(style="whitegrid")

def load_config():
    with open('backend/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_eda():
    config = load_config()
    data_dir = 'backend/data/'
    output_dir = 'backend/artifacts/figures/eda/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading datasets for EDA...")
    df_train = pd.read_csv(f"{data_dir}dataset_train.csv")
    df_shift = pd.read_csv(f"{data_dir}dataset_test_shift.csv")
    df_new = pd.read_csv(f"{data_dir}dataset_new_vaccine.csv")
    
    all_dfs = [df_train, df_shift, df_new]
    names = ['Train', 'Test Shift', 'New Vaccine']
    
    # Quick statistics
    for i, df in enumerate(all_dfs):
        print(f"\n--- {names[i]} Cohort ---")
        print(f"Shape: {df.shape}")
        print(f"Unique Subjects: {df['subject_id'].nunique()}")
        print(f"Low Responders: {df[df['day'] == 28]['low_responder_label'].sum()}")
        print(f"Mean Age: {df['age'].mean():.2f}")
        print(f"Mean BMI: {df['bmi'].mean():.2f}")
        print(f"Missing Values total: {df.isnull().sum().sum()}")

    # 2. Plot: Age Distribution Comparison
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(all_dfs):
        sns.kdeplot(df['age'], label=names[i], fill=True, alpha=0.3)
    plt.title("Age Distribution Across Cohorts")
    plt.legend()
    plt.savefig(f"{output_dir}age_distribution.png")
    plt.close()

    # 3. Plot: Antibody Titer at Day 28
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(all_dfs):
        sns.kdeplot(df[df['day'] == 28]['antibody_titer'], label=names[i], fill=True, alpha=0.3)
    plt.axvline(x=config['decision']['low_responder_threshold'], color='red', linestyle='--', label='Low Responder Cutoff')
    plt.title("Distribution of Antibody Titers at Day 28")
    plt.legend()
    plt.savefig(f"{output_dir}titer_distribution_d28.png")
    plt.close()

    # 4. Plot: Cytokine IL-6 Kinetics (Mean + Std)
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(all_dfs):
        sns.lineplot(data=df, x='day', y='cytokine_il6', label=names[i], marker='o')
    plt.title("Cytokine IL-6 Kinetics (Early Response)")
    plt.savefig(f"{output_dir}il6_kinetics.png")
    plt.close()

    # 5. Plot: Lymphocyte Kinetics
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(all_dfs):
        sns.lineplot(data=df, x='day', y='lymphocytes', label=names[i], marker='o')
    plt.title("Lymphocyte Kinetics (Adaptive Transition)")
    plt.savefig(f"{output_dir}lymphocyte_kinetics.png")
    plt.close()

    # 6. Plot: Covariate vs theta correlation (Train only)
    plt.figure(figsize=(8, 6))
    corr_cols = ['age', 'bmi', 'comorbidity_score', 'theta_activation', 'theta_prod', 'theta_decay']
    corr = df_train[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Covariate vs Parameter Correlation (Train Cohort)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}param_correlations.png")
    plt.close()

    print(f"\nEDA Figures saved to {output_dir}")

if __name__ == "__main__":
    run_eda()
