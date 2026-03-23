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
    
    # Safer Logic:
    if predicted_titer < threshold:
        # If the median is already below threshold, it's definitely not LOW risk.
        # It's HIGH if even the upper bound is struggling, otherwise MEDIUM.
        if upper_bound < threshold:
            return {
                'tier': 'HIGH',
                'prob_low_responder': 0.95,
                'action': tiers['high']['action'],
                'message': "High probability of inadequate response. Protective limit not reached."
            }
        else:
            return {
                'tier': 'MEDIUM',
                'prob_low_responder': 0.60,
                'action': tiers['medium']['action'],
                'message': "Borderline response. Median titer below protective limit."
            }
    
    # If median is above threshold, check the lower bound for uncertainty
    if lower_bound < threshold:
        return {
            'tier': 'MEDIUM',
            'prob_low_responder': 0.30,
            'action': tiers['medium']['action'],
            'message': "Median is protective, but uncertainty range falls below limit. Recommend monitoring."
        }
    else:
        return {
            'tier': 'LOW',
            'prob_low_responder': 0.05,
            'action': tiers['low']['action'],
            'message': "Optimal response. Low risk of falling below protective limit."
        }
