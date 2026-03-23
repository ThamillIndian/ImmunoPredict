from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import datetime
import json
import logging
from .schemas import PatientInput, PredictionResponse, RiskAssessment, PredictedParameters, TrajectoryPoint, BatchPredictionResponse
from .database import get_db, PredictionLog
from backend.models.pipeline import ImmunoPredictPipeline

router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global pipeline instance to be initialized in main.py
pipeline = None

def init_pipeline(config_path, model_path):
    global pipeline
    pipeline = ImmunoPredictPipeline(config_path, model_path)

def patient_to_df(patient: PatientInput):
    """
    Converts Pydantic PatientInput to the wide-feature DataFrame expected by the pipeline.
    """
    # 1. Create a dictionary with demographics
    data = {
        'age': [patient.age],
        'sex': [patient.sex],
        'bmi': [patient.bmi],
        'comorbidity_score': [patient.comorbidity_score]
    }
    
    # 2. Add biomarkers for each day (0, 1, 3, 7)
    # The pipeline expects columns like 'cytokine_il6_0', 'wbc_1', etc.
    biomarker_fields = ['cytokine_il6', 'cytokine_tnfa', 'cytokine_ifng', 'wbc', 'lymphocytes', 'neutrophils']
    
    # Initialize all with NaN to handle missing days
    for field in biomarker_fields:
        for day in [0, 1, 3, 7]:
            data[f"{field}_{day}"] = [float('nan')]
            
    # Fill in provided measurements
    for m in patient.measurements:
        if m.day in [0, 1, 3, 7]:
            data[f"cytokine_il6_{m.day}"] = [m.cytokine_il6]
            data[f"cytokine_tnfa_{m.day}"] = [m.cytokine_tnfa]
            data[f"cytokine_ifng_{m.day}"] = [m.cytokine_ifng]
            data[f"wbc_{m.day}"] = [m.wbc]
            data[f"lymphocytes_{m.day}"] = [m.lymphocytes]
            data[f"neutrophils_{m.day}"] = [m.neutrophils]
            
    # 3. Handle Imputation & Masks (Matching logic of pipeline/baseline)
    df = pd.DataFrame(data)
    
    # The scaler expects specific columns in specific order (52 features usually)
    # We should get the list of features from the pipeline's scaler
    feature_names = pipeline.scaler.get_feature_names_out()
    
    # Create final DF with all required columns, filling missing with 0 and adding masks
    final_data = {}
    for col in feature_names:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val):
                # Retrieve the training mean from the scaler to correctly map to a 0.0 scaled feature
                col_idx = list(feature_names).index(col)
                final_data[col] = [pipeline.scaler.mean_[col_idx]] 
            else:
                final_data[col] = [val]
        elif col.endswith("_observed"):
            base_col = col.replace("_observed", "")
            if base_col in df.columns:
                final_data[col] = [0.0 if pd.isna(df[base_col].iloc[0]) else 1.0]
            else:
                final_data[col] = [0.0]
        else:
            final_data[col] = [0.0]
            
    return pd.DataFrame(final_data)

@router.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput, db: Session = Depends(get_db)):
    logger.info(f"--- Incoming Prediction Request | Patient: {patient.patient_id} ---")
    logger.info(f"Demographics: Age={patient.age}, BMI={patient.bmi}, Vaccine={patient.vaccine_type}")
    logger.info(f"Measurements received: {len(patient.measurements)} day(s).")
    for m in patient.measurements:
        logger.info(f"  Day {m.day} -> WBC: {m.wbc}, IL-6: {m.cytokine_il6}, TNF-a: {m.cytokine_tnfa}, IFN-g: {m.cytokine_ifng}")

    if pipeline is None:
        logger.error("Pipeline is not initialized! Aborting prediction.")
        raise HTTPException(status_code=500, detail="Model pipeline not initialized")
    
    try:
        # 1. Format data
        logger.info("Converting patient payload to dataframe matching scalar features...")
        X = patient_to_df(patient)
        
        # 2. Run Pipeline
        logger.info("Executing Hybrid AI Pipeline (Scaling -> Encoder Neural Net -> ODE Monte Carlo -> Decision)...")
        res = pipeline.predict_patient(X, patient.vaccine_type)
        
        logger.info(f"Pipeline Execution Complete.")
        logger.info(f" -> Predicted Parameters (Theta): {res['predicted_theta']}")
        logger.info(f" -> Forecasted Day 28 Titer: {res['predicted_titer_28']:.2f} IU/mL")
        logger.info(f" -> Risk Decision: {res['risk_assessment']['tier']}")

        # 3. Format Response
        # Convert full trajectory to List[TrajectoryPoint]
        traj = []
        for i, day in enumerate(res['full_trajectory']['days']):
            traj.append(TrajectoryPoint(
                day=float(day),
                median=float(res['full_trajectory']['median'][i]),
                p5=float(res['full_trajectory']['p5'][i]),
                p95=float(res['full_trajectory']['p95'][i])
            ))
            
        response = PredictionResponse(
            patient_id=patient.patient_id,
            predicted_theta=PredictedParameters(**res['predicted_theta']),
            risk_assessment=RiskAssessment(
                tier=res['risk_assessment']['tier'],
                predicted_titer_28=res['predicted_titer_28'],
                confidence_interval_28=res['confidence_interval_28'],
                prob_low_responder=res['risk_assessment']['prob_low_responder'],
                recommendation=res['risk_assessment']['action']
            ),
            full_trajectory=traj,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        
        # 4. Log to DB
        log_entry = PredictionLog(
            patient_id=patient.patient_id,
            vaccine_type=patient.vaccine_type,
            predicted_titer_28=res['predicted_titer_28'],
            risk_tier=res['risk_assessment']['tier'],
            input_data=patient.model_dump(),
            full_output=response.model_dump()
        )
        db.add(log_entry)
        db.commit()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    return logs
