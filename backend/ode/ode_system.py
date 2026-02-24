import numpy as np
from scipy.integrate import solve_ivp

def vaccine_stimulation(t, vaccine_params):
    """
    Simulates the injection and early stimulation profile of the vaccine.
    s(t) = s0 * exp(-delta * t)
    """
    s0 = vaccine_params['s0']
    delta = vaccine_params['delta']
    return s0 * np.exp(-delta * t)

def immune_ode(t, y, theta_activation, theta_prod, theta_decay, kpd, vaccine_params):
    """
    3-state ODE model of immune response:
    dI/dt = s(t) - k_act * I
    dP/dt = k_act * I - kpd * P
    dA/dt = k_prod * P - k_decay * A
    
    States:
    y[0] = I (Innate activation)
    y[1] = P (Plasmablast/Effector response)
    y[2] = A (Antibody titer)
    """
    I, P, A = y
    
    s_t = vaccine_stimulation(t, vaccine_params)
    
    dI_dt = s_t - theta_activation * I
    dP_dt = theta_activation * I - kpd * P
    dA_dt = theta_prod * P - theta_decay * A
    
    return [dI_dt, dP_dt, dA_dt]

def simulate_trajectory(theta, vaccine_type, config):
    """
    Simulates the full trajectory for an individual with parameters theta.
    Returns t, I, P, A as a dictionary of arrays.
    """
    # Extract params from config
    vaccine_params = config['vaccines'][vaccine_type]
    kpd = config['ode']['kpd']
    y0 = config['ode']['initial_conditions']
    t_span = config['ode']['t_span']
    t_eval = np.linspace(t_span[0], t_span[1], config['ode']['t_eval_points'])
    
    # Extract theta components
    t_act = theta['activation']
    t_prod = theta['prod']
    t_decay = theta['decay']
    
    # Solve ODE
    sol = solve_ivp(
        immune_ode,
        t_span,
        y0,
        args=(t_act, t_prod, t_decay, kpd, vaccine_params),
        t_eval=t_eval,
        method=config['ode'].get('solver', 'RK45')
    )
    
    return {
        't': sol.t,
        'I': sol.y[0],
        'P': sol.y[1],
        'A': sol.y[2]
    }

def simulate_at_timepoints(theta, vaccine_type, timepoints, config):
    """
    Simulates and returns ODE states only at specific timepoints (e.g., [0, 1, 3, 7, 28, 90]).
    """
    vaccine_params = config['vaccines'][vaccine_type]
    kpd = config['ode']['kpd']
    y0 = config['ode']['initial_conditions']
    t_span = [0, max(timepoints)]
    
    t_act = theta['activation']
    t_prod = theta['prod']
    t_decay = theta['decay']
    
    sol = solve_ivp(
        immune_ode,
        t_span,
        y0,
        args=(t_act, t_prod, t_decay, kpd, vaccine_params),
        t_eval=timepoints,
        method=config['ode'].get('solver', 'RK45')
    )
    
    return {
        't': sol.t,
        'I': sol.y[0],
        'P': sol.y[1],
        'A': sol.y[2]
    }
