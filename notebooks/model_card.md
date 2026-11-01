# Model Card: FraudShield AI Dual Architecture

## Model Overview
- **Model Name:** FraudShield AI Dual Risk Engine
- **Model Architecture:**
  - **Classifier:** Class-Weighted Logistic Regression (optimizing Recall on rare fraud cases)
  - **Regressor:** LightGBM Regressor (predicting continuous 0-100 transaction risk score)
  - **Heuristic Blend:** Dynamic rule-based flags for location velocity, mismatch, and rapid failure spikes
- **Primary Use Case:** Real-time credit card transaction fraud screening and fintech risk assessment console.

## Performance Metrics
- **Fraud Classifier:**
  - Recall: 64.71%
  - ROC-AUC: 0.7148
  - PR-AUC: 0.2029
- **Risk Score Regressor:**
  - R² Score: 0.8008
  - MAE: 5.13
  - RMSE: 6.90

## Inputs & Features
- 55 total features (including numerical metrics, one-hot encoded categories, engineered behavioral features, and geographical distance calculations).
- Thresholds: Decision threshold at 0.50 probability; Suspicious risk threshold at 60.0.
