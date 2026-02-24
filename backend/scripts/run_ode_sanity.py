import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from backend.ode.ode_system import simulate_trajectory, simulate_at_timepoints
from backend.ode.monte_carlo import monte_carlo_trajectories

# Set aesthetic
sns.set_theme(style="whitegrid")

def load_config():
    with open('backend/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_sanity_checks():
    config = load_config()
    output_dir = 'backend/artifacts/figures/sanity/'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running ODE Sanity Checks...")

    # 1. Plot: A(t) for Strong vs Weak Responder
    plt.figure(figsize=(10, 6))
    
    strong_theta = {'activation': 0.7, 'prod': 0.4, 'decay': 0.05}
    weak_theta = {'activation': 0.3, 'prod': 0.15, 'decay': 0.15}
    
    strong_sol = simulate_trajectory(strong_theta, 'A', config)
    weak_sol = simulate_trajectory(weak_theta, 'A', config)
    
    plt.plot(strong_sol['t'], strong_sol['A'], label='Strong Responder', linewidth=2.5, color='dodgerblue')
    plt.plot(weak_sol['t'], weak_sol['A'], label='Weak Responder', linewidth=2.5, color='salmon')
    
    plt.axhline(y=config['decision']['low_responder_threshold'], color='gray', linestyle='--', label='Threshold')
    plt.title("Antibody Trajectory: Strong vs Weak Responder", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Antibody Level (A)", fontsize=12)
    plt.legend()
    plt.savefig(f"{output_dir}strong_vs_weak.png")
    plt.close()
    print("[OK] Saved strong_vs_weak.png")

    # 2. Plot: Full States (I, P, A) for one patient
    plt.figure(figsize=(10, 6))
    sol = simulate_trajectory(strong_theta, 'A', config)
    
    plt.plot(sol['t'], sol['I'], label='Innate (I)', color='purple', alpha=0.7)
    plt.plot(sol['t'], sol['P'], label='Plasmablast (P)', color='green', alpha=0.7)
    plt.plot(sol['t'], sol['A'], label='Antibody (A)', color='blue', linewidth=2)
    
    plt.title("Immune States Transition (I -> P -> A)", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("State Level", fontsize=12)
    plt.legend()
    plt.savefig(f"{output_dir}immune_states.png")
    plt.close()
    print("[OK] Saved immune_states.png")

    # 3. Plot: Vaccine A vs Vaccine B
    plt.figure(figsize=(10, 6))
    sol_A = simulate_trajectory(strong_theta, 'A', config)
    sol_B = simulate_trajectory(strong_theta, 'B', config)
    
    plt.plot(sol_A['t'], sol_A['A'], label='Vaccine A (Fast onset)', color='blue')
    plt.plot(sol_B['t'], sol_B['A'], label='Vaccine B (Slow onset, persistent)', color='orange')
    
    plt.title("Vaccine A vs Vaccine B Kinetics", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Antibody Level", fontsize=12)
    plt.legend()
    plt.savefig(f"{output_dir}vaccine_comparison.png")
    plt.close()
    print("[OK] Saved vaccine_comparison.png")

    # 4. Plot: Monte Carlo CI Bands
    print("Running Monte Carlo simulation for sanity check...")
    t_eval = np.linspace(0, 90, 50)
    stats = monte_carlo_trajectories(
        theta_means=[0.5, 0.3, 0.08], 
        theta_stds=[0.1, 0.05, 0.02], 
        vaccine_type='A', 
        timepoints=t_eval, 
        config=config
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(stats['t'], stats['A']['median'], label='Median A(t)', color='blue', linewidth=2)
    plt.fill_between(stats['t'], stats['A']['p05'], stats['A']['p95'], color='blue', alpha=0.2, label='90% CI Band')
    
    plt.title("Antibody Trajectory with Uncertainty (Monte Carlo)", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Antibody Level", fontsize=12)
    plt.legend()
    plt.savefig(f"{output_dir}monte_carlo_uncertainty.png")
    plt.close()
    print("[OK] Saved monte_carlo_uncertainty.png")

    print("\nPhase 1 Sanity Checks Completed Successfully.")

if __name__ == "__main__":
    run_sanity_checks()
