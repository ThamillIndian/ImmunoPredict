# ImmunoPredict — Final Implementation Plan

> **Project**: Individualized Modeling of Vaccine-Induced Immunity Using Mechanistic and AI Methods
> **Frontend**: Next.js | **Backend**: Python (FastAPI) | **Database**: SQLite
> **Training patients**: 500 (train) + 200 (test_shift) + 200 (new_vaccine)
> **Analysis**: Pure Python scripts (no Jupyter notebooks)

---

## Decisions Locked In

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Free θ parameters | 3 (activation, prod, decay) | Identifiable with limited titer observations |
| Fixed ODE parameter | kₚd (plasmablast decay) | Reduces underdetermined system |
| Biomarker → latent mapping | Yes (soft constraint) | Cytokines proxy for I(t), lymphocytes proxy for P(t) |
| Training loss | Multi-component (titer + biomarker proxy + KL reg) | Single titer loss too weak for 3 params |
| Responder label | Derived post-hoc from predicted A(28) | Not in training loss, avoids bias |
| Training approach | Two-stage (fit θ first, then train encoder) | Simpler, debuggable, stable for MVP |
| ODE solver | scipy.integrate.solve_ivp (RK45) | No GPU needed, reliable |
| Input representation | Flat vector + binary missingness mask | Simple, works with MLP |
| Uncertainty | Encoder predicts mean + std per θ, Monte Carlo ODE sampling | Gives confidence bands |
| Database | SQLite | Zero-config, sufficient for MVP |
| Frontend | Next.js | Dedicated frontend, impressive for panel |
| Genomic features | Out of scope (acknowledged as future work) | Depth over breadth |
| Booster simulation | Out of scope | Same reasoning |

---

## Final File Structure

```
e:\capstone\project\
│
├── backend/
│   ├── config.yaml                        # All parameters, thresholds, distributions
│   ├── requirements.txt                   # Python dependencies
│   │
│   ├── ode/
│   │   ├── __init__.py
│   │   ├── ode_system.py                  # 3-state ODE (I, P, A) + stimulation function
│   │   └── monte_carlo.py                 # Sample θ → run N simulations → CI bands
│   │
│   ├── data_gen/
│   │   ├── __init__.py
│   │   ├── config_loader.py               # Load and validate config.yaml
│   │   ├── population.py                  # Generate demographics + individual θ values
│   │   ├── biomarkers.py                  # Convert ODE states → realistic biomarker measurements
│   │   └── generate_datasets.py           # Main script → outputs 3 CSVs
│   │
│   ├── data/
│   │   ├── dataset_train.csv              # 500 patients × 7 days = 3,500 rows
│   │   ├── dataset_test_shift.csv         # 200 patients, older/sicker demographics
│   │   └── dataset_new_vaccine.csv        # 200 patients, Vaccine B kinetics
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                     # MLP: features → θ (mean + std)
│   │   ├── preprocessing.py               # Long→wide pivot, scaling, imputation, mask
│   │   ├── dataset.py                     # PyTorch Dataset class
│   │   ├── decision.py                    # Risk tier logic (LOW/MED/HIGH → action)
│   │   └── pipeline.py                    # Full inference: input → preprocess → encode → ODE → decide
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── baseline.py                    # XGBoost + simple MLP baseline
│   │   ├── stage1_fit_theta.py            # Per-patient θ fitting via scipy.optimize
│   │   ├── stage2_train_encoder.py        # Train MLP encoder on fitted θ
│   │   └── evaluate.py                    # All metrics, figures, comparison tables
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                        # FastAPI app + startup + CORS
│   │   ├── routes.py                      # Endpoint definitions
│   │   ├── schemas.py                     # Pydantic request/response models
│   │   └── database.py                    # SQLite connection + CRUD operations
│   │
│   ├── artifacts/
│   │   ├── encoder.pt                     # Trained encoder weights
│   │   ├── scaler.pkl                     # Fitted StandardScaler
│   │   ├── config.json                    # Runtime model config
│   │   ├── fitted_theta.csv               # Stage 1 output
│   │   ├── training_log.json              # Loss curves
│   │   ├── baseline/                      # Baseline model files
│   │   │   ├── xgboost_model.pkl
│   │   │   └── baseline_metrics.json
│   │   └── figures/                       # All evaluation plots (PNG)
│   │
│   ├── scripts/                           # Analysis scripts (replaces notebooks)
│   │   ├── run_ode_sanity.py              # ODE verification plots
│   │   ├── run_eda.py                     # Dataset exploration + distributions
│   │   ├── run_baseline.py                # Run baseline and save metrics
│   │   ├── run_theta_fitting.py           # Run Stage 1 and save results
│   │   ├── run_encoder_training.py        # Run Stage 2 and save model
│   │   └── run_evaluation.py              # Generate all final figures + tables
│   │
│   └── db/
│       └── predictions.db                 # SQLite (auto-created at runtime)
│
├── frontend/                              # Next.js application
│   ├── package.json
│   ├── next.config.js
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.js
│   │   │   ├── page.js                    # Landing / home
│   │   │   ├── predict/
│   │   │   │   └── page.js                # Single patient prediction form + results
│   │   │   ├── batch/
│   │   │   │   └── page.js                # CSV upload + batch results table
│   │   │   ├── history/
│   │   │   │   └── page.js                # Past prediction lookup
│   │   │   └── dashboard/
│   │   │       └── page.js                # Overview stats + charts
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── PatientForm.jsx            # Input form for biomarkers
│   │   │   ├── TrajectoryChart.jsx         # A(t) curve with CI bands (recharts/chart.js)
│   │   │   ├── RiskBadge.jsx              # Color-coded risk tier display
│   │   │   ├── ParameterGauge.jsx         # θ parameter visualization
│   │   │   ├── BatchResultsTable.jsx      # Sortable patient results table
│   │   │   └── StatsCards.jsx             # Summary statistics cards
│   │   ├── lib/
│   │   │   └── api.js                     # Fetch wrapper for backend API calls
│   │   └── styles/
│   │       └── globals.css
│   └── ...
│
├── docs/
│   ├── data_dictionary.md                 # Every column explained
│   └── architecture_diagram.png
│
├── plan.md                                # Original plan (kept for reference)
├── plan1.md                               # This file
├── README.md
└── .gitignore
```

---

## Phase 0 — Project Setup & Configuration

### What to build
1. Create the `backend/` directory and all subdirectories
2. Create `backend/requirements.txt` with all dependencies
3. Create `backend/config.yaml` — the single source of truth for all parameters
4. Create all `__init__.py` files
5. Create `.gitignore` (ignore `__pycache__`, `*.pyc`, `db/`, `artifacts/`, `data/*.csv`)

### config.yaml detailed contents

```yaml
# Population distributions
population:
  train:
    n_subjects: 500
    age: { mean: 45, std: 15, min: 18, max: 80 }
    bmi: { mean: 25, std: 4, min: 16, max: 35 }
    sex_ratio: 0.5                        # 50% male
    comorbidity_weights: [0.4, 0.3, 0.2, 0.1]  # P(score=0,1,2,3)
    noise_level: 0.1                      # coefficient of variation for biomarkers
    missingness_rate: 0.05                # 5% missing values

  test_shift:
    n_subjects: 200
    age: { mean: 62, std: 12, min: 30, max: 80 }
    bmi: { mean: 28, std: 5, min: 18, max: 35 }
    sex_ratio: 0.55
    comorbidity_weights: [0.2, 0.25, 0.3, 0.25]
    noise_level: 0.15
    missingness_rate: 0.12

  new_vaccine:
    n_subjects: 200
    age: { mean: 45, std: 15, min: 18, max: 80 }
    bmi: { mean: 25, std: 4, min: 16, max: 35 }
    sex_ratio: 0.5
    comorbidity_weights: [0.4, 0.3, 0.2, 0.1]
    noise_level: 0.1
    missingness_rate: 0.05

# Timepoints measured
timepoints: [0, 1, 3, 7, 14, 28, 90]
early_days: [0, 1, 3, 7]                 # encoder input days
titer_days: [14, 28, 90]                  # days where titer is observed

# Vaccine stimulation profiles
vaccines:
  A:
    s0: 5.0                               # initial stimulation strength
    delta: 0.8                            # stimulation decay rate
  B:
    s0: 3.0                               # slower onset
    delta: 0.4                            # longer persistence

# ODE parameters
ode:
  kpd: 0.3                               # plasmablast decay (FIXED for all patients)
  initial_conditions: [0.0, 0.0, 0.0]    # I(0), P(0), A(0)
  solver: RK45
  t_span: [0, 100]
  t_eval_points: 200                     # resolution of ODE output

# θ parameter ranges (for generation + regularization bounds)
theta:
  activation:
    population_mean: 0.5
    population_std: 0.15
    bounds: [0.1, 1.5]
  prod:
    population_mean: 0.3
    population_std: 0.1
    bounds: [0.05, 1.0]
  decay:
    population_mean: 0.08
    population_std: 0.03
    bounds: [0.01, 0.3]

# θ modifiers based on covariates
theta_modifiers:
  age_effect: -0.005                     # per year above 45, reduce activation by this
  bmi_effect: -0.008                     # per unit above 25, reduce prod by this
  comorbidity_effect: 0.015              # per score point, increase decay by this
  sex_female_boost: 0.05                 # females get slight activation boost

# Biomarker generation
biomarkers:
  cytokine_il6:
    baseline: 2.0
    scale_from: I                        # derived from I(t)
    scale_factor: 15.0
  cytokine_tnfa:
    baseline: 1.5
    scale_from: I
    scale_factor: 8.0
  cytokine_ifng:
    baseline: 0.8
    scale_from: IP                       # mix of I(t) and P(t)
    scale_factor_I: 3.0
    scale_factor_P: 5.0
  wbc:
    baseline: 7.0
    scale_from: IP
    scale_factor_I: 2.0
    scale_factor_P: 1.5
  lymphocytes:
    baseline: 2.0
    scale_from: P
    scale_factor: 2.5
  neutrophils:
    baseline: 4.0
    scale_from: I
    scale_factor: 3.0

# Decision thresholds
decision:
  low_responder_threshold: 80.0          # A(28) below this = low responder
  risk_tiers:
    high:
      prob_low_responder_min: 0.7
      confidence_min: 0.6
      action: FOLLOW_UP
    medium:
      action: TEST
    low:
      prob_low_responder_max: 0.2
      confidence_min: 0.6
      action: MONITOR

# Training hyperparameters
training:
  encoder:
    hidden_layers: [128, 64, 32]
    dropout: 0.2
    learning_rate: 0.001
    batch_size: 32
    epochs: 200
    early_stopping_patience: 20
    train_val_split: 0.8
  monte_carlo:
    n_samples: 100                       # number of θ samples for CI bands
  stage1:
    n_restarts: 5                        # random restarts for θ fitting
    optimizer: L-BFGS-B
    fit_quality_threshold: 0.95          # filter patients with R² below this
```

### Verification
- `config.yaml` loads without errors via `yaml.safe_load()`
- All paths exist
- `pip install -r requirements.txt` succeeds

---

## Phase 1 — ODE Simulator

### What to build

#### `backend/ode/ode_system.py`
```
Functions:
├── vaccine_stimulation(t, vaccine_params) → float
│     s(t) = s0 * exp(-delta * t)
│
├── immune_ode(t, y, theta, vaccine_params, kpd) → [dI, dP, dA]
│     y = [I, P, A]
│     dI/dt = s(t) - theta_activation * I
│     dP/dt = theta_activation * I - kpd * P
│     dA/dt = theta_prod * P - theta_decay * A
│
├── simulate_trajectory(theta, vaccine_type, config) → dict
│     Uses solve_ivp(RK45)
│     Returns: { t: array, I: array, P: array, A: array }
│
└── simulate_at_timepoints(theta, vaccine_type, timepoints, config) → dict
      Returns I, P, A only at specified days
```

#### `backend/ode/monte_carlo.py`
```
Functions:
├── sample_theta(mean, std, n_samples) → array of shape (n, 3)
│     Clip samples to valid bounds
│
└── monte_carlo_trajectories(theta_mean, theta_std, vaccine_type, config) → dict
      Run ODE n_samples times
      Returns: {
        median_trajectory: array,
        ci_lower: array (5th percentile),
        ci_upper: array (95th percentile),
        all_trajectories: array (n_samples × timepoints)
      }
```

### Verification: `backend/scripts/run_ode_sanity.py`
This script generates and saves:
1. Plot: A(t) for a "strong responder" θ = (0.7, 0.4, 0.05) → expect peak ~200 at Day 28
2. Plot: A(t) for a "weak responder" θ = (0.3, 0.15, 0.15) → expect peak ~50
3. Plot: I(t), P(t), A(t) all together for one patient → verify timing
4. Plot: Vary each θ param independently → confirm directional effects
5. Plot: Vaccine A vs Vaccine B with same θ → verify different kinetics
6. Plot: Monte Carlo CI bands for one θ (mean + std) → verify bands are reasonable

All plots saved to `backend/artifacts/figures/sanity/`

Run: `cd backend && python -m scripts.run_ode_sanity`

---

## Phase 2 — Synthetic Dataset Generation

### What to build

#### `backend/data_gen/config_loader.py`
```
Functions:
└── load_config(path="config.yaml") → dict
      Load YAML, validate required keys exist, return config dict
```

#### `backend/data_gen/population.py`
```
Functions:
├── generate_demographics(cohort_config, n_subjects) → DataFrame
│     Columns: subject_id, age, sex, bmi, comorbidity_score
│     Distributions from config
│
└── generate_theta(demographics_df, theta_config, modifiers) → DataFrame
      Adds columns: theta_activation, theta_prod, theta_decay
      Base θ sampled from population distributions
      Modified by age, BMI, comorbidity, sex effects
      Clipped to valid bounds
```

#### `backend/data_gen/biomarkers.py`
```
Functions:
├── generate_biomarkers(ode_states, day, config) → dict
│     Takes I(t), P(t), A(t) at a specific day
│     Returns: {cytokine_il6, cytokine_tnfa, cytokine_ifng,
│               wbc, lymphocytes, neutrophils}
│     Each = baseline + scale_factor * ODE_state + noise
│
├── compute_derived_scores(biomarkers) → dict
│     innate_score = normalize(il6 + tnfa + neutrophils)
│     adaptive_score = normalize(ifng + lymphocytes)
│
├── apply_noise(values, noise_level) → values with Gaussian noise
│     noise ~ N(0, noise_level * value)
│     Clip to >= 0
│
└── apply_missingness(df, rate) → df with random NaN insertions
      Only in biomarker columns, not outcomes
      MCAR (missing completely at random)
```

#### `backend/data_gen/generate_datasets.py`
```
Main script flow:
1. Load config
2. For each cohort (train, test_shift, new_vaccine):
   a. Generate demographics (population.py)
   b. Generate individual θ values (population.py)
   c. For each patient:
      - Simulate ODE trajectory (ode/ode_system.py)
      - For each timepoint:
        - Generate biomarker measurements (biomarkers.py)
        - If day in [14, 28, 90]: record antibody_titer = A(day) + noise
        - If day in [0, 1, 3, 7]: antibody_titer = NaN
   d. Assemble long-format DataFrame
   e. Compute low_responder_label (A(28) < threshold from config)
   f. Apply missingness
   g. Save CSV

Output columns per row:
  subject_id, cohort, vaccine_type, day,
  age, sex, bmi, comorbidity_score,
  cytokine_il6, cytokine_tnfa, cytokine_ifng,
  wbc, lymphocytes, neutrophils,
  innate_score, adaptive_score,
  antibody_titer, low_responder_label,
  theta_activation, theta_prod, theta_decay
```

### Verification: `backend/scripts/run_eda.py`
1. Load all 3 CSVs, print shapes (expect 3500, 1400, 1400 rows)
2. Print column dtypes, null counts
3. Verify no negative biomarker values
4. Verify antibody_titer is NaN for early days, numeric for later days
5. Plot: age distributions per cohort (train vs test_shift should differ)
6. Plot: comorbidity distributions per cohort
7. Plot: antibody_titer at Day 28 histogram per cohort
8. Plot: low_responder_label proportions per cohort
9. Plot: cytokine_il6 over time (mean ± std) → should peak Day 1
10. Plot: lymphocytes over time → should rise to Day 7
11. Print missingness rates per column per cohort
12. Verify test_shift has more missingness than train

All plots saved to `backend/artifacts/figures/eda/`

Run: `cd backend && python -m scripts.run_eda`

---

## Phase 3 — Baseline ML Model

### What to build

#### `backend/train/baseline.py`
```
Functions:
├── prepare_baseline_features(df, early_days, config) → X, y
│     Pivot long → wide (one row per patient)
│     Features: all biomarkers at each early day (flattened)
│               + age, sex, bmi, comorbidity_score
│               + vaccine_type one-hot
│     Fill NaN with 0, add _observed mask columns
│     y = antibody_titer at Day 28
│
├── train_xgboost(X_train, y_train) → model, metrics
│     5-fold cross-validation
│     Return best model + {MAE, RMSE, R²}
│
├── train_mlp_baseline(X_train, y_train) → model, metrics
│     Simple sklearn MLPRegressor
│     Same CV setup
│
├── evaluate_responder_classification(y_true, y_pred, threshold) → metrics
│     Threshold predicted titer → predicted label
│     Return {AUC, precision, recall, F1}
│
└── save_baseline_results(model, metrics, path)
      Save model .pkl + metrics .json
```

### Verification: `backend/scripts/run_baseline.py`
1. Train XGBoost + MLP on training data
2. Print MAE, RMSE, R², AUC for each
3. Plot: feature importance (top 15 features from XGBoost)
4. Plot: predicted vs true A(28) scatter
5. Save all to `backend/artifacts/baseline/`

Run: `cd backend && python -m scripts.run_baseline`

---

## Phase 4 — Stage 1: Per-Patient θ Fitting

### What to build

#### `backend/train/stage1_fit_theta.py`
```
Functions:
├── objective(theta, patient_titer_data, patient_biomarker_data, config) → float
│     Run ODE with given theta
│     loss = MSE(A_pred at titer_days, A_true at titer_days)
│          + 0.1 * MSE(I_pred at early_days, cytokine_proxy at early_days)
│     return loss
│
├── fit_single_patient(patient_data, config) → dict
│     Run scipy.optimize.minimize (L-BFGS-B)
│     5 random restarts from different initial θ
│     Keep best result
│     Return {theta_activation, theta_prod, theta_decay, fit_quality}
│
├── fit_all_patients(dataset, config) → DataFrame
│     Use joblib.Parallel for speed
│     Return fitted_theta.csv content
│
└── filter_poor_fits(fitted_df, threshold) → DataFrame
      Remove patients where fit_quality < threshold
      Print how many removed
```

### Verification: `backend/scripts/run_theta_fitting.py`
1. Run fitting on all 500 training patients
2. Print: fit quality statistics (mean, median, min, max R²)
3. Print: number of patients filtered out
4. Plot: fitted θ_activation vs ground-truth θ_activation (scatter, expect R² > 0.8)
5. Plot: same for θ_prod and θ_decay
6. Plot: fit quality histogram
7. Save `backend/artifacts/fitted_theta.csv`

Run: `cd backend && python -m scripts.run_theta_fitting`

---

## Phase 5 — Stage 2: Encoder Training

### What to build

#### `backend/models/encoder.py`
```
class ImmuneEncoder(nn.Module):
    Input:  N features (flattened biomarkers + masks + covariates)
    Layers: Linear(N, 128) → BatchNorm → ReLU → Dropout(0.2)
            Linear(128, 64) → BatchNorm → ReLU → Dropout(0.2)
            Linear(64, 32) → BatchNorm → ReLU → Dropout(0.2)
            Linear(32, 6)  → split into 3 means + 3 log_stds

    Methods:
    ├── forward(x) → (means, stds)
    │     means = output[:3]
    │     stds = softplus(output[3:6]) + 1e-4   # ensure positive
    │
    └── sample_theta(means, stds, n=100) → tensor of shape (n, 3)
          Reparameterization trick: θ = μ + σ * ε, ε ~ N(0,1)
```

#### `backend/models/preprocessing.py`
```
Functions:
├── pivot_to_wide(df, early_days) → DataFrame
│     One row per patient
│     Columns: biomarker_dayX for each biomarker and each early day
│
├── add_missingness_mask(df) → DataFrame
│     For each biomarker column with NaN:
│       add column_observed = 0 if NaN, 1 if present
│     Fill original NaN with 0
│
├── fit_scaler(df, feature_cols) → StandardScaler
│     Fit on training data
│     Save to artifacts/scaler.pkl
│
├── transform_features(df, scaler, feature_cols) → numpy array
│     Apply fitted scaler
│
└── get_feature_columns(config) → list of column names
      Deterministic list of all feature columns in correct order
```

#### `backend/models/dataset.py`
```
class ImmuneDataset(torch.utils.data.Dataset):
    __init__(features_array, theta_array)
    __len__() → int
    __getitem__(idx) → (features_tensor, theta_tensor)
```

#### `backend/train/stage2_train_encoder.py`
```
Functions:
├── gaussian_nll_loss(theta_pred_mean, theta_pred_std, theta_true) → loss
│     NLL = 0.5 * (log(σ²) + (θ_true - μ)² / σ²)
│     Summed over 3 parameters
│
├── kl_regularization(pred_mean, pred_std, pop_mean, pop_std) → loss
│     KL divergence between predicted and population prior
│
├── train_epoch(model, dataloader, optimizer) → avg_loss
│
├── validate(model, dataloader) → avg_loss
│
└── train_encoder(config) → trained model path
      Load preprocessed data + fitted theta
      Split 80/20 train/val
      Training loop with:
        - Gaussian NLL loss
        - KL regularization (weight 0.01)
        - Adam optimizer with ReduceLROnPlateau
        - Early stopping (patience 20)
      Save best model → artifacts/encoder.pt
      Save training log → artifacts/training_log.json
      Save scaler → artifacts/scaler.pkl
      Save feature config → artifacts/config.json
```

### Verification: `backend/scripts/run_encoder_training.py`
1. Run full training
2. Plot: training loss vs validation loss curves (expect convergence, no divergence)
3. Plot: predicted θ vs fitted θ (scatter, 3 subplots)
4. Print: MAE for each θ parameter
5. Test: predicted std is higher for elderly/comorbid patients (correlation check)
6. Test: simulate ODE with predicted θ → compute A(28) error vs true
7. Save everything to artifacts/

Run: `cd backend && python -m scripts.run_encoder_training`

---

## Phase 6 — Full Pipeline Integration

### What to build

#### `backend/models/decision.py`
```
Functions:
├── compute_risk_assessment(trajectories_at_28, config) → dict
│     Input: array of A(28) values from Monte Carlo samples
│     Compute:
│       median_a28 = median of samples
│       prob_low = fraction of samples below threshold
│       ci_width = 95th - 5th percentile
│       confidence = clip(1 - ci_width/median_a28, 0, 1)
│     Apply rules:
│       if prob_low > 0.7 and confidence > 0.6:
│         risk_tier = HIGH, action = FOLLOW_UP
│       elif prob_low < 0.2 and confidence > 0.6:
│         risk_tier = LOW, action = MONITOR
│       else:
│         risk_tier = MEDIUM, action = TEST
│     Generate explanation string
│     Return: { risk_tier, confidence, predicted_a28, prob_low_responder,
│               recommended_action, explanation }
│
└── generate_explanation(risk_tier, prob_low, confidence, theta_mean) → str
      Human-readable text explaining the prediction
      References which θ parameters are high/low
```

#### `backend/ode/monte_carlo.py` (already defined in Phase 1, finalize here)

#### `backend/models/pipeline.py`
```
class PredictionPipeline:
    __init__(model_path, scaler_path, config_path)
      Load encoder, scaler, config once

    predict(patient_data: dict) → PredictionResult
      1. Validate input
      2. Create DataFrame from patient_data
      3. pivot_to_wide → add_mask → scale
      4. Encoder forward pass → θ_mean, θ_std
      5. monte_carlo_trajectories(θ_mean, θ_std) → trajectory + CI
      6. compute_risk_assessment(trajectories_at_28) → risk
      7. Assemble full result

    predict_batch(patients: list[dict]) → list[PredictionResult]
      Run predict for each patient

PredictionResult:
    patient_id: str
    predicted_parameters: { activation: {mean, std}, prod: {mean, std}, decay: {mean, std} }
    predicted_trajectory: [ {day, antibody, ci_lower, ci_upper}, ... ]
    risk_assessment: { risk_tier, confidence, predicted_a28, prob_low_responder,
                       recommended_action, explanation }
    timestamp: str
```

### Verification
1. Run pipeline on 10 known patients with known ground-truth
2. Verify strong responder → LOW risk
3. Verify weak responder → HIGH risk
4. Verify ambiguous case → MEDIUM risk
5. Verify CI bands exist and are wider for uncertain cases
6. Verify JSON output matches expected schema

Run: `cd backend && python -m scripts.run_evaluation --quick-test`

---

## Phase 7 — Evaluation & All Figures

### What to build

#### `backend/train/evaluate.py`
```
Functions:
├── evaluate_titer_prediction(pipeline, dataset) → metrics dict
│     MAE, RMSE, R² on A(28)
│
├── evaluate_responder_classification(pipeline, dataset, threshold) → metrics dict
│     AUC, precision, recall, F1
│
├── evaluate_theta_recovery(pipeline, dataset) → metrics dict
│     MAE between predicted θ and ground-truth θ per parameter
│
├── compare_with_baseline(hybrid_metrics, baseline_metrics) → comparison table
│
├── cross_vaccine_evaluation(pipeline, vaccine_b_dataset) → metrics dict
│     Same metrics, Vaccine B data
│
├── cohort_shift_evaluation(pipeline, shifted_dataset) → metrics dict
│     Same metrics, shifted test set
│
└── generate_all_figures(results, output_dir)
      Saves all PNGs
```

### Figures to generate (saved to `backend/artifacts/figures/`)

| # | Filename | Content |
|---|----------|---------|
| 1 | `pred_vs_true_a28.png` | Scatter plot + R² annotation |
| 2 | `example_trajectories_strong.png` | 3 strong responders with CI bands |
| 3 | `example_trajectories_weak.png` | 3 weak responders with CI bands |
| 4 | `uncertainty_comparison.png` | High-confidence vs low-confidence side by side |
| 5 | `generalization_vaccine_b.png` | Performance on Vaccine B cohort |
| 6 | `cohort_shift_performance.png` | Train vs shifted test metrics comparison |
| 7 | `theta_recovery_scatter.png` | 3 subplots: pred θ vs true θ |
| 8 | `baseline_vs_hybrid.png` | Bar chart comparing all metrics |
| 9 | `risk_distribution.png` | Pie/bar of LOW/MED/HIGH per cohort |
| 10 | `feature_importance.png` | From XGBoost baseline |

### Verification: `backend/scripts/run_evaluation.py`
Run: `cd backend && python -m scripts.run_evaluation`

---

## Phase 8 — FastAPI Backend

### What to build

#### `backend/api/schemas.py`
```
Pydantic models:

class Measurement(BaseModel):
    day: int                              # 0, 1, 3, or 7
    cytokine_il6: float | None
    cytokine_tnfa: float | None
    cytokine_ifng: float | None
    wbc: float | None
    lymphocytes: float | None
    neutrophils: float | None

class PatientInput(BaseModel):
    patient_id: str
    vaccine_type: Literal["A", "B"]
    age: int                              # 18–80
    sex: Literal[0, 1]
    bmi: float                            # 16–35
    comorbidity_score: Literal[0, 1, 2, 3]
    measurements: list[Measurement]       # 1–4 entries (Day 0,1,3,7)

class TrajectoryPoint(BaseModel):
    day: int
    antibody: float
    ci_lower: float
    ci_upper: float

class ParameterEstimate(BaseModel):
    mean: float
    std: float

class PredictedParameters(BaseModel):
    theta_activation: ParameterEstimate
    theta_prod: ParameterEstimate
    theta_decay: ParameterEstimate

class RiskAssessment(BaseModel):
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    confidence: float
    predicted_a28: float
    prob_low_responder: float
    recommended_action: Literal["MONITOR", "TEST", "FOLLOW_UP"]
    explanation: str

class PredictionResponse(BaseModel):
    patient_id: str
    predicted_parameters: PredictedParameters
    predicted_trajectory: list[TrajectoryPoint]
    risk_assessment: RiskAssessment
    timestamp: str

class BatchResponse(BaseModel):
    total_patients: int
    results: list[PredictionResponse]
    summary: dict

class HistoryEntry(BaseModel):
    run_id: str
    timestamp: str
    predicted_a28: float
    risk_tier: str
    recommended_action: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str
```

#### `backend/api/database.py`
```
Functions:
├── init_db() → create predictions table if not exists
├── save_prediction(prediction_response) → run_id
├── get_history(patient_id) → list[HistoryEntry]
├── get_all_predictions(limit, offset) → list[HistoryEntry]
└── get_stats() → { total, risk_distribution, avg_confidence }
```

#### `backend/api/routes.py`
```
Endpoints:

GET  /api/v1/health
  → HealthResponse

POST /api/v1/predict
  Body: PatientInput
  → PredictionResponse
  Side effect: save to SQLite

POST /api/v1/batch_predict
  Body: multipart/form-data (CSV file)
  → BatchResponse
  Side effect: save each to SQLite

GET  /api/v1/history/{patient_id}
  → { patient_id, predictions: list[HistoryEntry] }

GET  /api/v1/stats
  → { total_predictions, risk_distribution, avg_confidence, predictions_today }
```

#### `backend/api/main.py`
```
FastAPI app:
  - on_startup: load pipeline (encoder + scaler + config), init DB
  - CORS middleware (allow frontend origin)
  - Include routes
  - Exception handlers (422 validation, 500 internal)

Run: uvicorn backend.api.main:app --reload --port 8000
```

### Verification
Test all endpoints manually or via script:
1. `GET /api/v1/health` → 200 with model_loaded=true
2. `POST /api/v1/predict` with valid patient → 200 with full response
3. `POST /api/v1/predict` with missing required field → 422
4. `POST /api/v1/predict` with out-of-range values → 422
5. `POST /api/v1/batch_predict` with small CSV → 200 with array
6. `GET /api/v1/history/P001` → 200 with saved prediction
7. `GET /api/v1/stats` → 200 with correct counts

Run: `cd backend && uvicorn api.main:app --reload`
Then: `cd backend && python -m scripts.test_api` (simple requests-based test script)

---

## Phase 9 — Next.js Frontend

### What to build

#### Pages

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Home / Landing | Project overview, navigation to features |
| `/predict` | Single Prediction | Patient form → trajectory chart + risk card |
| `/batch` | Batch Prediction | CSV upload → results table + population summary |
| `/history` | Prediction History | Look up past predictions by patient ID |
| `/dashboard` | Overview Dashboard | Aggregate stats, risk distribution charts |

#### Key Components

| Component | What it does |
|-----------|-------------|
| `Navbar` | Navigation across pages |
| `PatientForm` | Input form with all fields + validation |
| `TrajectoryChart` | Interactive A(t) line chart with CI shading (Recharts or Chart.js) |
| `RiskBadge` | Color-coded pill: 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH |
| `ParameterGauge` | Visual bars showing θ values with labels |
| `BatchResultsTable` | Sortable, filterable table with risk column |
| `StatsCards` | Cards showing total predictions, risk breakdown, avg confidence |
| `FileUpload` | CSV drag-and-drop uploader |

#### API Integration: `src/lib/api.js`
```
Functions:
├── predictSingle(patientData) → POST /api/v1/predict
├── predictBatch(csvFile) → POST /api/v1/batch_predict
├── getHistory(patientId) → GET /api/v1/history/{patientId}
├── getStats() → GET /api/v1/stats
└── healthCheck() → GET /api/v1/health
```

### Design Requirements
- Dark theme with medical/biotech aesthetic
- Smooth animations on data load
- Responsive (desktop-first but mobile-friendly)
- Loading states with skeleton UI
- Error handling with toast notifications

### Verification
1. Start backend: `cd backend && uvicorn api.main:app --port 8000`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to each page — no errors
4. Submit single prediction → chart renders + risk badge appears
5. Upload CSV → table populates with results
6. Check history → shows saved predictions
7. Dashboard → stats cards + charts render

---

## Phase 10 — Documentation & Polish

### What to build
1. `README.md` — project overview, setup instructions, screenshots
2. `docs/data_dictionary.md` — every column defined with units and ranges
3. `docs/architecture_diagram.png` — system architecture visual
4. Clean up all scripts, add docstrings
5. Add error handling throughout
6. Final manual end-to-end test

---

## Build Order Summary

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3 (parallel) ─┐
                                  →  Phase 4  →  Phase 5  → Phase 6  → Phase 7
                                                                      ↓
                                                               Phase 8  →  Phase 9  →  Phase 10
```

Each phase is self-contained with its own verification step. No phase starts until its dependencies are verified.
