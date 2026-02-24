# Individualized Modeling of Vaccine-Induced Immunity Using Mechanistic and AI Methods

## 1. Project Overview

### Goal

Build a hybrid mechanistic (ODE/QSP) + AI system that models
individualized vaccine-induced immune responses using synthetic datasets
for MVP development.

### Output

-   Personalized immune trajectory A(t)
-   Low-responder risk classification
-   Decision-support recommendations (monitor / test / follow-up)

------------------------------------------------------------------------

## 2. System Architecture

### Offline Training Pipeline

1.  Generate synthetic datasets
2.  Preprocess data (scaling, missing handling)
3.  Train AI encoder to infer individual parameters (θᵢ)
4.  Simulate immune ODE model
5.  Optimize using antibody titers
6.  Validate on distribution-shift and new vaccine cohorts
7.  Save trained model

### Online Inference Pipeline

1.  Input patient biomarkers + metadata
2.  Preprocess input
3.  Infer θᵢ
4.  Simulate antibody trajectory
5.  Apply decision rules
6.  Return trajectory + risk tier

------------------------------------------------------------------------

## 3. Synthetic Dataset Design

### Datasets to Generate

1.  dataset_train.csv
2.  dataset_test_shift.csv
3.  dataset_new_vaccine.csv

### Timepoints

0, 1, 3, 7, 14, 28, 90

### Core Columns

-   subject_id
-   cohort
-   vaccine_type
-   day
-   age
-   sex
-   bmi
-   comorbidity_score
-   cytokine_il6
-   cytokine_tnfa
-   cytokine_ifng
-   wbc
-   lymphocytes
-   neutrophils
-   innate_score
-   adaptive_score
-   antibody_titer
-   low_responder_label
-   theta_activation
-   theta_expansion
-   theta_decay
-   theta_prod

------------------------------------------------------------------------

## 4. Mechanistic Immune Model

States: - I(t): Innate activation - P(t): Plasmablast response - A(t):
Antibody titer

Equations: - dI/dt = s(t) − kᵢ·I - dP/dt = kₚ·I − kₚd·P - dA/dt = kₐ·P −
kₐd·A

Individual parameters θᵢ vary per subject.

------------------------------------------------------------------------

## 5. AI Model

### Encoder

Input: - Early biomarkers - Clinical covariates - Vaccine type

Output: - θᵢ (mean + std for uncertainty)

Loss: - MSE between predicted and observed antibody titers -
Regularization on parameter ranges

------------------------------------------------------------------------

## 6. Decision Support Layer

Rules: - Low predicted titer + low uncertainty → suggest testing - High
uncertainty → recommend measurement - Strong response → routine
monitoring

Outputs: - risk_tier - confidence - recommended_action

------------------------------------------------------------------------

## 7. Tech Stack

### Modeling

-   Python
-   NumPy
-   Pandas
-   PyTorch
-   SciPy / torchdiffeq

### Backend

-   FastAPI
-   PostgreSQL
-   Docker

### Frontend

-   Streamlit (MVP)

------------------------------------------------------------------------

## 8. Development Roadmap

### Phase 1

Generate synthetic datasets

### Phase 2

Baseline ML model

### Phase 3

Mechanistic ODE simulator

### Phase 4

Hybrid model integration

### Phase 5

API + Dashboard

### Phase 6

Validation and reporting

------------------------------------------------------------------------

## 9. Dataset Generation Prompt

Use structured ODE-based immune simulation to generate biologically
plausible longitudinal datasets with noise, missingness, and population
shifts.

------------------------------------------------------------------------

End of Plan
