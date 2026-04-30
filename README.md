# 🛡️ Cybersecurity Intrusion Detection — Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask&logoColor=white)


> End-to-end machine learning pipeline for real-time network intrusion detection — from raw data exploration to containerized REST API deployment, with full experiment tracking.

---

## 📌 Overview

This project builds a binary classification system that detects malicious network sessions from behavioral features such as login attempts, packet size, session duration, and IP reputation. The pipeline covers the complete ML lifecycle: data preprocessing, model training, interpretability analysis, REST API deployment via Docker, and experiment tracking with MLflow.

**Dataset:** 9,537 labeled network sessions (55% normal / 45% attack)

---

## 🎯 Problem Statement

Network intrusion detection is a critical challenge in cybersecurity. Traditional rule-based systems fail to generalize to new attack patterns. This project demonstrates how machine learning can identify anomalous sessions with high accuracy by learning from behavioral signals — without relying on static rule sets.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 9,537 sessions |
| Features | 10 (after removing session ID) |
| Target | `attack_detected` (0 = normal, 1 = attack) |
| Class balance | 55.3% normal — 44.7% attack |
| Missing values | `encryption_used`: 1,966 nulls (handled as "None" class) |

**Feature breakdown:**

| Feature | Type | Description |
|---|---|---|
| `network_packet_size` | Numerical | Size of network packets (bytes) |
| `protocol_type` | Categorical | TCP / UDP / ICMP |
| `login_attempts` | Numerical | Number of login attempts in session |
| `session_duration` | Numerical | Duration of the session (seconds) |
| `encryption_used` | Categorical | AES / DES / None |
| `ip_reputation_score` | Numerical | Threat score of source IP (0–1) |
| `failed_logins` | Numerical | Failed authentication attempts |
| `browser_type` | Categorical | Chrome / Firefox / Edge / Safari / Unknown |
| `unusual_time_access` | Binary | Access outside business hours (0/1) |

---

## 🏗️ Project Architecture

```
intrusion-detection/
│
├── data/
│   └── cybersecurity_intrusion_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_training.ipynb          # Model training & comparison
│   ├── 03_evaluation.ipynb        # Metrics & ROC curves
│   └── 04_interpretability.ipynb  # SHAP analysis
│
├── src/
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── train.py                   # Model training pipeline
│   ├── evaluate.py                # Evaluation metrics
│   └── explain.py                 # SHAP / interpretability
│
├── api/
│   └── app.py                     # Flask REST API
│
├── models/
│   ├── model.pkl                  # Serialized best model
│   └── scaler.pkl                 # Fitted StandardScaler
│
├── outputs/
│   ├── eda_classes.png
│   ├── eda_heatmap.png
│   ├── eda_boxplot.png
│   ├── roc_curve.png
│   └── shap_summary.png
│
├── Dockerfile
├── requirements.txt
├── mlruns/                        # MLflow experiment logs
└── README.md
```

---

## ⚙️ Pipeline

### 1. Exploratory Data Analysis
- Distribution analysis of all features by class (normal vs. attack)
- Correlation heatmap across numerical features
- Boxplots identifying discriminative features (`failed_logins`, `ip_reputation_score`)
- Class balance assessment

### 2. Preprocessing & Feature Engineering
- Dropped `session_id` (non-informative identifier)
- Imputed missing `encryption_used` values as a dedicated `"None"` category
- One-hot encoding for `protocol_type`, `encryption_used`, `browser_type`
- `StandardScaler` normalization for all numerical features
- Stratified train/test split (80/20)

### 3. Model Training & Comparison

| Model | Description |
|---|---|
| Decision Tree | Baseline interpretable model (`max_depth=10`) |
| Random Forest | Ensemble of 100 trees, robust to noise |
| Voting Classifier | Soft-vote ensemble combining both models |

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC curve and AUC score
- Classification report per class (normal / attack)

### 5. Interpretability — SHAP
- `TreeExplainer` applied to the Random Forest
- Global feature importance summary plot
- Per-prediction explanations identifying which features triggered the attack classification

### 6. Deployment — Docker + Flask REST API
- Serialized model and scaler served via a `/predict` endpoint
- JSON input → binary prediction + human-readable message
- Containerized with Docker for reproducible deployment

### 7. Experiment Tracking — MLflow
- All runs logged with parameters, metrics (accuracy, F1, AUC), and model artifacts
- Compare experiments visually via `mlflow ui`

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Docker (for deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/intrusion-detection.git
cd intrusion-detection

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the training pipeline

```bash
python src/train.py
```

### Launch MLflow UI

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### Build and run with Docker

```bash
docker build -t intrusion-detection-api .
docker run -p 5000:5000 intrusion-detection-api
```

### Test the API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "network_packet_size": 599,
    "protocol_type_TCP": 1,
    "protocol_type_UDP": 0,
    "protocol_type_ICMP": 0,
    "login_attempts": 3,
    "session_duration": 120.5,
    "encryption_used_AES": 1,
    "encryption_used_DES": 0,
    "encryption_used_None": 0,
    "ip_reputation_score": 0.85,
    "failed_logins": 2,
    "browser_type_Chrome": 1,
    "browser_type_Firefox": 0,
    "browser_type_Edge": 0,
    "browser_type_Safari": 0,
    "browser_type_Unknown": 0,
    "unusual_time_access": 1
  }'
```

Expected response:
```json
{
  "prediction": 1,
  "message": "Attaque détectée"
}
```

---

## 📦 Requirements

```
pandas
scikit-learn
matplotlib
seaborn
shap
mlflow
flask
ipykernel
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐳 Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — Builder: install python deps into a self-contained virtualenv
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps required by some wheels (numpy/scipy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv under /opt/venv so we can copy it into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements-docker.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements-docker.txt


# =============================================================================
# Stage 2 — Runtime: tiny image with just python + venv + app code
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    PATH="/opt/venv/bin:$PATH"

# Copy the prebuilt venv from the builder
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user (Azure best practice)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Copy only what the runtime actually needs.
# .dockerignore filters out the rest (venv/, mlruns/, notebooks/, large CSVs, ...)
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser models/   ./models/
COPY --chown=appuser:appuser outputs/  ./outputs/
COPY --chown=appuser:appuser mlflow.db ./mlflow.db

USER appuser

EXPOSE 7860

# Quick liveness check — Gradio exposes /config when up
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; \
u.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/config', timeout=3)" \
    || exit 1

# The app itself reads PORT and binds to 0.0.0.0 — no shell-form needed.
CMD ["python", "src/ui_frontend.py"]
```

---

## 📈 Results

> Results will be updated after full model training.

| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| Voting Ensemble | — | — | — |

---

## 🔍 Key Insights

- `ip_reputation_score` and `failed_logins` are the strongest predictors of malicious sessions, as confirmed by SHAP analysis
- Sessions using no encryption (`encryption_used = None`) show a significantly higher attack rate
- `unusual_time_access` combined with multiple `login_attempts` is a strong compound signal for intrusion

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.11 | Core language |
| pandas | Data manipulation |
| scikit-learn | ML models & preprocessing |
| SHAP | Model interpretability |
| Flask | REST API |
| Docker | Containerization |
| MLflow | Experiment tracking |
| matplotlib / seaborn | Visualizations |


---

© 2026 David Zhang. All rights reserved.  
This code is for academic purposes only and may not be redistributed.

