import numpy as np

def apply_noise(values, noise_level):
    """
    Applies Gaussian noise proportional to the signal value.
    noise ~ N(0, noise_level * value)
    """
    noise = np.random.normal(0, noise_level * np.abs(values))
    return np.clip(values + noise, 0, None)

def generate_biomarkers(ode_states, noise_level, config):
    """
    Converts ODE latent states (I, P, A) into observable biomarkers.
    """
    bm_cfg = config['biomarkers']
    I = ode_states['I']
    P = ode_states['P']
    
    # helper for IP mix
    def mix_ip(cfg, I_val, P_val):
        return cfg['baseline'] + cfg['scale_factor_I'] * I_val + cfg['scale_factor_P'] * P_val

    results = {
        'cytokine_il6': bm_cfg['cytokine_il6']['baseline'] + bm_cfg['cytokine_il6']['scale_factor'] * I,
        'cytokine_tnfa': bm_cfg['cytokine_tnfa']['baseline'] + bm_cfg['cytokine_tnfa']['scale_factor'] * I,
        'cytokine_ifng': mix_ip(bm_cfg['cytokine_ifng'], I, P),
        'wbc': mix_ip(bm_cfg['wbc'], I, P),
        'lymphocytes': bm_cfg['lymphocytes']['baseline'] + bm_cfg['lymphocytes']['scale_factor'] * P,
        'neutrophils': bm_cfg['neutrophils']['baseline'] + bm_cfg['neutrophils']['scale_factor'] * I
    }
    
    # Apply noise to all
    for key in results:
        results[key] = apply_noise(results[key], noise_level)
        
    return results

def compute_derived_scores(biomarkers_df):
    """
    Computes innate_score and adaptive_score as weighted/normalized composites.
    """
    # Innate: driven by IL6, TNFa, Neutrophils
    i_score = (biomarkers_df['cytokine_il6'] + 
               biomarkers_df['cytokine_tnfa'] + 
               biomarkers_df['neutrophils']) / 3.0
    
    # Adaptive: driven by IFNg, Lymphocytes
    a_score = (biomarkers_df['cytokine_ifng'] + 
               biomarkers_df['lymphocytes']) / 2.0
               
    return i_score, a_score

def apply_missingness(df, rate):
    """
    Randomly sets biomarker values to NaN based on missingness_rate.
    Outcomes (antibody_titer, labels) and demographics are NOT affected.
    """
    biomarker_cols = [
        'cytokine_il6', 'cytokine_tnfa', 'cytokine_ifng',
        'wbc', 'lymphocytes', 'neutrophils'
    ]
    
    # Generate mask
    mask = np.random.rand(*df[biomarker_cols].shape) < rate
    
    result_df = df.copy()
    # Apply mask using Pandas .mask for robust assignment
    result_df[biomarker_cols] = result_df[biomarker_cols].mask(mask, np.nan)
    
    return result_df
