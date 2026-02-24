import numpy as np

def get_risk_tier(predicted_titer, confidence_interval, config):
    """
    Categorizes a patient into Risk Tiers based on predicted titer 
    and the uncertainty (confidence interval).
    
    Tiers:
    - HIGH RISK: Titer likely < threshold
    - LOW RISK: Titer likely > threshold
    - MEDIUM/UNCERTAIN: Either close to threshold or high uncertainty
    """
    threshold = config['decision']['low_responder_threshold']
    tiers = config['decision']['risk_tiers']
    
    lower_bound, upper_bound = confidence_interval
    
    # Simple logic based on threshold crossing
    if upper_bound < threshold:
        return {
            'tier': 'HIGH',
            'prob_low_responder': 0.95, # Symbolic for now
            'action': tiers['high']['action'],
            'message': "High risk of low vaccine response. Clinical follow-up recommended."
        }
    elif lower_bound > threshold:
        return {
            'tier': 'LOW',
            'prob_low_responder': 0.05,
            'action': tiers['low']['action'],
            'message': "Low risk. Standard monitoring sufficient."
        }
    else:
        return {
            'tier': 'MEDIUM',
            'prob_low_responder': 0.50,
            'action': tiers['medium']['action'],
            'message': "Borderline response or high uncertainty. Recommendation: Serology test at Day 28."
        }
