import requests
import json

def test_prediction():
    url = "http://127.0.0.1:8000/api/predict"
    
    # Sample Subject 1 (Strong Responder) Data
    payload = {
        "patient_id": "TEST_001",
        "vaccine_type": "A",
        "age": 45,
        "sex": 0,
        "bmi": 24.5,
        "comorbidity_score": 0,
        "measurements": [
            {
                "day": 0,
                "cytokine_il6": 2.1,
                "cytokine_tnfa": 1.5,
                "cytokine_ifng": 0.8,
                "wbc": 7.2,
                "lymphocytes": 2.1,
                "neutrophils": 4.2
            },
            {
                "day": 1,
                "cytokine_il6": 150.5, # Strong early immune spike
                "cytokine_tnfa": 12.0,
                "cytokine_ifng": 5.0,
                "wbc": 10.5,
                "lymphocytes": 2.5,
                "neutrophils": 8.0
            }
        ]
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Successfully received clinical risk report:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_prediction()
