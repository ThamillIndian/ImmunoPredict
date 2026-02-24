from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class Measurement(BaseModel):
    day: int = Field(..., description="Day of measurement (0, 1, 3, or 7)")
    cytokine_il6: float | None = None
    cytokine_tnfa: float | None = None
    cytokine_ifng: float | None = None
    wbc: float | None = None
    lymphocytes: float | None = None
    neutrophils: float | None = None

class PatientInput(BaseModel):
    patient_id: str
    vaccine_type: Literal["A", "B"]
    age: int = Field(..., ge=18, le=90)
    sex: int = Field(..., description="0 for Male, 1 for Female")
    bmi: float = Field(..., ge=15, le=45)
    comorbidity_score: int = Field(..., ge=0, le=3)
    # The pipeline expects a dataframe. We'll handle the conversion in the route.
    measurements: List[Measurement]

class TrajectoryPoint(BaseModel):
    day: float
    median: float
    p5: float
    p95: float

class PredictedParameters(BaseModel):
    activation: float
    prod: float
    decay: float

class RiskAssessment(BaseModel):
    tier: str
    predicted_titer_28: float
    confidence_interval_28: List[float]
    prob_low_responder: float
    recommendation: str

class PredictionResponse(BaseModel):
    patient_id: str
    predicted_theta: PredictedParameters
    risk_assessment: RiskAssessment
    full_trajectory: List[TrajectoryPoint]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total: int
    results: List[PredictionResponse]
