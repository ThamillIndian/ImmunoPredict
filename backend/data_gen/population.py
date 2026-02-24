import numpy as np
import pandas as pd

def generate_demographics(cohort_config, n_subjects):
    """
    Generates synthetic patient demographics based on cohort-specific distributions.
    """
    # Age: Normal distribution clipped
    age_cfg = cohort_config['age']
    ages = np.random.normal(age_cfg['mean'], age_cfg['std'], n_subjects)
    ages = np.clip(ages, age_cfg['min'], age_cfg['max']).astype(int)
    
    # BMI: Normal distribution clipped
    bmi_cfg = cohort_config['bmi']
    bmis = np.random.normal(bmi_cfg['mean'], bmi_cfg['std'], n_subjects)
    bmis = np.clip(bmis, bmi_cfg['min'], bmi_cfg['max'])
    
    # Sex: Binary (0=Female, 1=Male)
    sex_ratio = cohort_config['sex_ratio']
    sexes = np.random.choice([0, 1], size=n_subjects, p=[1 - sex_ratio, sex_ratio])
    
    # Comorbidity Score: Categorical (0, 1, 2, 3)
    weights = cohort_config['comorbidity_weights']
    comorb_scores = np.random.choice([0, 1, 2, 3], size=n_subjects, p=weights)
    
    df = pd.DataFrame({
        'subject_id': np.arange(1, n_subjects + 1),
        'age': ages,
        'sex': sexes,
        'bmi': bmis,
        'comorbidity_score': comorb_scores
    })
    
    return df

def generate_theta(demographics_df, theta_config, modifiers):
    """
    Maps demographics to individual theta parameters with biological logic + noise.
    theta_activation = base_act + age_mod + sex_mod + noise
    theta_prod = base_prod + bmi_mod + noise
    theta_decay = base_decay + comorb_mod + noise
    """
    n = len(demographics_df)
    
    # 1. Sample base parameters from population distributions
    act_base = np.random.normal(theta_config['activation']['population_mean'], 
                                theta_config['activation']['population_std'], n)
    prod_base = np.random.normal(theta_config['prod']['population_mean'], 
                                 theta_config['prod']['population_std'], n)
    decay_base = np.random.normal(theta_config['decay']['population_mean'], 
                                  theta_config['decay']['population_std'], n)
    
    # 2. Apply covariate modifiers
    # Age reduces activation
    age_eff = (demographics_df['age'] - 45) * modifiers['age_effect']
    
    # BMI reduces production
    bmi_eff = (demographics_df['bmi'] - 25) * modifiers['bmi_effect']
    
    # Comorbidity increases decay (faster waning)
    comorb_eff = demographics_df['comorbidity_score'] * modifiers['comorbidity_effect']
    
    # Females (sex=0) get a boost in activation
    sex_eff = (1 - demographics_df['sex']) * modifiers['sex_female_boost']
    
    # 3. Combine
    df = demographics_df.copy()
    df['theta_activation'] = act_base + age_eff + sex_eff
    df['theta_prod'] = prod_base + bmi_eff
    df['theta_decay'] = decay_base + comorb_eff
    
    # 4. Clip to hard biological bounds
    df['theta_activation'] = df['theta_activation'].clip(theta_config['activation']['bounds'][0], 
                                                        theta_config['activation']['bounds'][1])
    df['theta_prod'] = df['theta_prod'].clip(theta_config['prod']['bounds'][0], 
                                            theta_config['prod']['bounds'][1])
    df['theta_decay'] = df['theta_decay'].clip(theta_config['decay']['bounds'][0], 
                                              theta_config['decay']['bounds'][1])
    
    return df
