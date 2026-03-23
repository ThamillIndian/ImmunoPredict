import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Measurement {
  day: number;
  cytokine_il6?: number;
  cytokine_tnfa?: number;
  cytokine_ifng?: number;
  wbc?: number;
  lymphocytes?: number;
  neutrophils?: number;
}

export interface PatientPayload {
  patient_id: string;
  vaccine_type: "A" | "B";
  age: number;
  sex: number;
  bmi: number;
  comorbidity_score: number;
  measurements: Measurement[];
}

export interface PredictionResult {
  patient_id: string;
  predicted_theta: {
    activation: number;
    prod: number;
    decay: number;
  };
  risk_assessment: {
    tier: string;
    predicted_titer_28: number;
    confidence_interval_28: number[];
    prob_low_responder: number;
    recommendation: string;
  };
  full_trajectory: {
    day: number;
    median: number;
    p5: number;
    p95: number;
  }[];
  timestamp: string;
}

export const submitPrediction = async (data: PatientPayload): Promise<PredictionResult> => {
  const response = await api.post('/predict', data);
  return response.data;
};

export const fetchHistory = async () => {
  const response = await api.get('/history');
  return response.data;
};
