"""Quick test: verify config.yaml loads correctly."""
import yaml
import sys
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("CONFIG LOAD TEST")
    print("=" * 50)
    print(f"[OK] Config loaded successfully")
    print(f"[OK] Train subjects: {config['population']['train']['n_subjects']}")
    print(f"[OK] Test shift subjects: {config['population']['test_shift']['n_subjects']}")
    print(f"[OK] New vaccine subjects: {config['population']['new_vaccine']['n_subjects']}")
    print(f"[OK] Vaccine A s0: {config['vaccines']['A']['s0']}")
    print(f"[OK] Vaccine B s0: {config['vaccines']['B']['s0']}")
    print(f"[OK] ODE kpd (fixed): {config['ode']['kpd']}")
    print(f"[OK] Theta activation bounds: {config['theta']['activation']['bounds']}")
    print(f"[OK] Encoder layers: {config['training']['encoder']['hidden_layers']}")
    print(f"[OK] Low responder threshold: {config['decision']['low_responder_threshold']}")
    print(f"[OK] Timepoints: {config['timepoints']}")
    print("=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)

except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)
