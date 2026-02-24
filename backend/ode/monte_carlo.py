import numpy as np
from .ode_system import simulate_at_timepoints

def sample_theta(means, stds, n_samples, config):
    """
    Samples n_samples of theta vectors using the predicted means and stds.
    Respects the hard bounds defined in config.
    """
    # Sample from Gaussian
    samples = np.random.normal(means, stds, (n_samples, 3))
    
    # Extract bounds from config
    bounds = [
        config['theta']['activation']['bounds'],
        config['theta']['prod']['bounds'],
        config['theta']['decay']['bounds']
    ]
    
    # Clip to bounds
    for i in range(3):
        samples[:, i] = np.clip(samples[:, i], bounds[i][0], bounds[i][1])
        
    return samples

def monte_carlo_trajectories(theta_means, theta_stds, vaccine_type, timepoints, config):
    """
    Runs Monte Carlo simulations and returns aggregated trajectory statistics.
    """
    n_samples = config['training']['monte_carlo']['n_samples']
    
    # Sample thetas
    theta_list = sample_theta(theta_means, theta_stds, n_samples, config)
    
    # Prepare results storage
    all_A = np.zeros((n_samples, len(timepoints)))
    all_I = np.zeros((n_samples, len(timepoints)))
    all_P = np.zeros((n_samples, len(timepoints)))
    
    # Run simulations
    for i in range(n_samples):
        theta_dict = {
            'activation': theta_list[i, 0],
            'prod': theta_list[i, 1],
            'decay': theta_list[i, 2]
        }
        sol = simulate_at_timepoints(theta_dict, vaccine_type, timepoints, config)
        all_I[i, :] = sol['I']
        all_P[i, :] = sol['P']
        all_A[i, :] = sol['A']
        
    # Aggregate (median and 5th/95th percentiles)
    stats = {
        't': timepoints,
        'A': {
            'median': np.median(all_A, axis=0),
            'p05': np.percentile(all_A, 5, axis=0),
            'p95': np.percentile(all_A, 95, axis=0),
        },
        'I': {
            'median': np.median(all_I, axis=0),
            'p05': np.percentile(all_I, 5, axis=0),
            'p95': np.percentile(all_I, 95, axis=0),
        },
        'P': {
            'median': np.median(all_P, axis=0),
            'p05': np.percentile(all_P, 5, axis=0),
            'p95': np.percentile(all_P, 95, axis=0),
        }
    }
    
    return stats
