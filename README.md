# 🛡️ Black-Box Adversarial Intrusion Detection System

**EnCryptoGuard** is an adversarial machine-learning research project that combines network intrusion detection with **black-box and white-box evasion attacks**.

The system trains and serves three intrusion-detection models—**Random Forest, LightGBM, and a PyTorch feed-forward neural network (FNN)**—then evaluates how an attacker can manipulate malicious traffic so that the models classify it as benign.

The repository also includes a Streamlit interface for traffic detection, attack experiments, model comparison, and analytics.

> **Research / defensive-security use:** This project is intended for controlled experimentation, robustness testing, and defensive IDS research. Do not use it against systems or traffic you do not own or have authorization to test.

## 🎯 Project Goals

The project explores two questions:

1. How accurately can an ensemble detect malicious network traffic?
2. How robust is that detector when an attacker can query or differentiate through the model?

The implementation covers:

- Network-flow preprocessing and cleaning
- Binary benign/attack classification
- Random Forest training
- LightGBM training
- PyTorch FNN training with class weighting
- Three-model probability-averaged ensemble inference
- Domain-aware black-box genetic attacks
- White-box FGSM and PGD evasion experiments
- Streamlit dashboard and attack-analysis UI

## 🧠 System Architecture

```text
                    Network Flow Data
                           │
                           ▼
                ┌──────────────────────┐
                │ Preprocessing        │
                │ merge → clean →      │
                │ sample / align       │
                └──────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Random Forest   LightGBM      FNN
              │            │            │
              └────────────┼────────────┘
                           ▼
                 Probability Average
                           │
                    threshold = 0.5
                           │
                    Benign / Attack
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   Black-Box GA Attack                White-Box Attacks
   query-only ensemble               FGSM + PGD on FNN
```

## 📊 Dataset Pipeline

The preprocessing scripts implement a staged network-flow pipeline.

### 1. Merge raw CSV files

`preprocessing/merge_and_clean.py` reads CSV files from `dataset/raw/`, concatenates them, strips column-name whitespace, and converts labels to binary values:

- `BENIGN` → `0`
- everything else → `1`

The merged data is written to `dataset/processed/master_dataset.csv`.

### 2. Clean the dataset

`preprocessing/clean_dataset.py`:

- replaces `±inf` with `NaN`
- removes rows containing missing values
- removes duplicate rows
- writes `dataset/processed/cleaned_dataset.csv`

### 3. Create the FNN/attack working subset

`preprocessing/create_subset.py` samples **500,000 rows** with `random_state=42` and writes `dataset/processed/fnn_dataset.csv`.

## 🤖 Detection Models

### Random Forest

Implemented in `training/train_rf.py`.

- `RandomForestClassifier`
- **100 trees**
- `random_state=42`
- `n_jobs=-1`
- 80/20 stratified train-test split

Saved as `models/rf_model.pkl`.

### LightGBM

Implemented in `training/train_lgbm.py`.

- `LGBMClassifier`
- **100 estimators**
- `random_state=42`
- 80/20 stratified train-test split

Saved as `models/lgbm_model.pkl`.

### Feed-Forward Neural Network

The production FNN is defined in `core/model_registry.py` and trained by `training/train_fnn_v2.py`.

Architecture:

```text
Input
  ↓
Linear → 256 → ReLU → Dropout(0.3)
  ↓
Linear → 128 → ReLU → Dropout(0.3)
  ↓
Linear → 64 → ReLU
  ↓
Linear → 1
  ↓
Sigmoid probability
```

Training configuration:

- PyTorch
- `BCEWithLogitsLoss`
- positive-class weighting for imbalance
- Adam, learning rate **0.001**
- **50 epochs**
- DataLoader batch size **2048**
- `StandardScaler` fitted on training data
- best-loss checkpoint

Artifacts:

```text
models/fnn_v2_model.pth
models/fnn_v2_best.pth
models/fnn_scaler.pkl
```

## 🧮 Ensemble Detection

The production predictor is implemented in `gui/components/predictor.py`.

For each uploaded sample:

1. The centralized preprocessor aligns the input to the training feature schema.
2. Random Forest predicts an attack probability.
3. LightGBM predicts an attack probability.
4. FNN predicts an attack probability after scaling.
5. The three probabilities are averaged.

```text
ensemble_probability =
    (RF_probability + LightGBM_probability + FNN_probability) / 3
```

Configured threshold:

- probability ≥ **0.5** → Attack
- probability < **0.5** → Benign

The preprocessor also strips column whitespace, removes an optional `Label`, fills missing expected columns with zero, drops unexpected columns, enforces feature order, replaces NaN/inf, and clips selected traffic quantities to non-negative values.

## 🧬 Black-Box Genetic Attack

The black-box attack is implemented under `attacks/blackbox/`.

The attacker does **not** access gradients or model weights. It interacts with an `EnsembleOracle` exposing RF, LightGBM, FNN, and averaged probabilities.

### Oracle

`attacks/blackbox/oracle.py`:

- lazy-loads the three models
- counts oracle queries
- enforces a hard query budget
- returns per-model probabilities
- returns the ensemble probability
- uses the same 0.5 detection threshold

### Genetic algorithm

`attacks/blackbox/ga_engine.py` uses DEAP and implements:

- population size: **100**
- maximum generations: **80**
- elite size: **5**
- crossover probability: **0.7**
- mutation probability: **0.30**
- per-gene mutation probability: **0.10**
- adaptive mutation scale
- elitism
- early stop when ensemble probability falls below **0.45**
- default query budget: **10,000**

### Fitness objective

```text
fitness =
    (1 - ensemble_probability)
    - 0.05 × normalized_perturbation
    - 0.10 × realism_penalty
```

Perturbation is measured relative to the original attack seed rather than a zero vector.

### Domain-aware constraints

`attacks/blackbox/constraints.py` attempts to preserve realistic traffic by enforcing:

- non-negative packet/byte/duration/IAT/count-style features
- soft p1–p99 data bounds
- integer handling for flag-like features
- byte/packet consistency
- a p5–p95-based realism penalty

### Command line

```bash
python -m attacks.blackbox.run_blackbox_attack
python -m attacks.blackbox.run_blackbox_attack --seed 42
python -m attacks.blackbox.run_blackbox_attack --budget 10000 --runs 5
python -m attacks.blackbox.run_blackbox_attack --out results/blackbox.json
```

Results can include success/failure, generation, query count, final ensemble probability, perturbation norm, and realism penalty.

## ⚡ White-Box Attacks

### FGSM

Implemented in `attacks/whitebox/fgsm.py`.

Configured epsilon values:

```text
0.01, 0.03, 0.05, 0.07, 0.10
```

The current implementation targets evasion of **attack samples** toward the benign class and reports overall accuracy degradation plus attack-class evasion rate.

### PGD

Implemented in `attacks/whitebox/pgd.py`.

Configuration:

- ε = **0.05**
- α = **0.01**
- **20 iterations**
- **5 random restarts**
- L∞ projection
- only attack-labelled samples are perturbed
- targeted objective toward benign classification

The repository also contains an earlier `fgsm_experiment.py` that saves experiment results to a CSV.

## 🖥️ Streamlit Interface

Run:

```bash
streamlit run app.py
```

The application exposes:

| Page | Purpose |
|---|---|
| **Dashboard** | Model accuracy comparison and system status |
| **Traffic Detection** | Upload CSVs and run ensemble detection |
| **White-Box Attacks** | UI entry points for FGSM/PGD |
| **Black-Box Attacks** | Configure and execute the genetic attack |
| **Analytics** | Reserved for evaluation/attack analytics |

The traffic-detection page reports attack/benign counts, average threat score, per-model agreement, a threat-probability histogram, and the highest-risk samples. The black-box page reports queries, probability, perturbation norm, realism penalty, generation trends, per-model probabilities, and top feature perturbations.

## 📁 Repository Structure

```text
BlackBox-Adversarial-Intrusion-Detection-System/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── artifacts/
│   ├── X_test.pkl
│   └── y_test.pkl
├── attacks/
│   ├── blackbox/
│   │   ├── constraints.py
│   │   ├── fitness.py
│   │   ├── ga_engine.py
│   │   ├── genome.py
│   │   ├── oracle.py
│   │   └── run_blackbox_attack.py
│   └── whitebox/
│       ├── fgsm.py
│       ├── fgsm_experiment.py
│       └── pgd.py
├── core/
│   ├── config.py
│   ├── model_registry.py
│   └── preprocessor.py
├── evaluation/
├── gui/
├── models/
├── preprocessing/
└── training/
```

## 🚀 Installation

### Requirements

The repository uses:

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- LightGBM
- PyTorch
- joblib
- Plotly
- Matplotlib
- tqdm
- DEAP

Install them with:

```bash
pip install -r requirements.txt
```

### Start the application

```bash
streamlit run app.py
```

### Rebuild the preprocessing pipeline

From the repository root:

```bash
python preprocessing/merge_and_clean.py
python preprocessing/clean_dataset.py
python preprocessing/create_subset.py
```

### Train the models

```bash
python training/train_rf.py
python training/train_lgbm.py
python training/train_fnn_v2.py
```

## 📈 Recorded Dashboard Metrics

The dashboard currently displays:

| Model | Displayed Accuracy |
|---|---:|
| Random Forest | **99.85%** |
| LightGBM | **99.91%** |
| FNN | **98.19%** |

These values are **hard-coded dashboard values**, not metrics dynamically recomputed by the dashboard. They should therefore be treated as project-recorded figures rather than a fresh benchmark run during documentation.

The repository contains trained binary artifacts and test tensors, but those binaries cannot be reliably inspected as text through the GitHub text-file interface. Re-run the training/evaluation scripts when reproducible benchmark tables are required.

## ⚠️ Current Limitations

- Several evaluation scripts are placeholders or experimental.
- The Analytics page is currently a placeholder.
- The White-Box Streamlit page is not yet fully wired to the FGSM/PGD implementations.
- Dashboard accuracy values are hard-coded.
- `evaluation/art_blackbox.py` is an experimental ART/HopSkipJump script rather than part of the main UI flow.
- Large binary model/test artifacts are committed directly to the repository.
- The Streamlit black-box page mutates global GA configuration for its run.
- Experiment results are not consistently persisted as a single benchmark dataset.
- The traffic constraints are heuristic and should be validated against protocol-aware network semantics before being treated as realistic packet-flow transformations.

## 🔬 Recommended Extensions

- Add stratified/repeated evaluation across multiple dataset splits.
- Persist standardized clean, FGSM, PGD, and black-box metrics.
- Report precision, recall, F1, ROC-AUC, PR-AUC, and attack-class recall together.
- Measure black-box success rate versus query budget.
- Measure perturbation magnitude versus evasion rate.
- Add ablations for realism and perturbation penalties.
- Run multiple random seeds and report confidence intervals.
- Replace heuristic constraints with protocol-aware transformations.
- Complete the Analytics and White-Box UI pages.
- Add automated tests for preprocessing, feature schema alignment, and attack constraints.
- Consider external artifact storage for large trained models.

## 🛡️ Responsible Security Use

This repository is appropriate for controlled:

- IDS robustness research
- adversarial ML experimentation
- defensive model validation
- cybersecurity education
- benchmarking of query-based and gradient-based evasion

Use only on data, models, and systems you own or are explicitly authorized to test.

## 👤 Author

**Harshit Garg**

GitHub: [Harshit765G4](https://github.com/Harshit765G4)

---

Built as an experimental adversarial IDS research platform combining network-flow classification, ensemble detection, and adversarial robustness testing.
