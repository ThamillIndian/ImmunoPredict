import yaml
import os

def load_config(config_path=None):
    """
    Loads the central config.yaml file.
    Default search path: backend/config.yaml or ../config.yaml relative to caller.
    """
    if config_path is None:
        # Try a few common paths
        paths_to_try = [
            'backend/config.yaml',
            'config.yaml',
            '../config.yaml',
            '../../config.yaml'
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                config_path = p
                break
    
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.yaml. Please provide a valid path.")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
