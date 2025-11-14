# MLflow Lab: California Housing Price Prediction

## Lab Overview

A comprehensive machine learning lab demonstrating MLflow's capabilities for experiment tracking, model management, and deployment using the California Housing dataset for price classification.

This lab builds an end-to-end ML pipeline that:

- Predicts whether California houses are "high-value" (median price ≥ $350k)
- Trains and compares 4 different classification algorithms
Uses MLflow for experiment tracking, hyperparameter optimization, and model registry
- Implements staging workflows and REST API deployment

**Dataset**: 20,640 housing districts, 11 features (8 original + 3 engineered)  
**Models Trained**: Gradient Boosting, Random Forest, AdaBoost, Logistic Regression  
**Best Model**: Random Forest (AUC: 0.9530)

---

## Setup Instructions

```bash
# Install dependencies
pip install mlflow scikit-learn xgboost hyperopt pandas numpy seaborn matplotlib

# Launch MLflow UI
mlflow ui --port=5000

# Serve production model
mlflow models serve --env-manager=local -m models:/housing_classifier/Production -p 5001
```

## Changes Made to the Lab

- **Dataset**: Implemented California Housing dataset (20,640 samples, 8 features) from sklearn.datasets instead of CSV files
- **Feature Engineering**: Created 3 derived features (rooms_per_person, bedrooms_per_room, people_per_house)
- **Primary Model**: Used Gradient Boosting Classifier instead of Random Forest as baseline
- **Model Comparison**: Trained and registered 4 different algorithms (Gradient Boosting, Random Forest, AdaBoost, Logistic Regression)
- **Hyperparameter Optimization**: Configured 6 hyperparameters with 20 Hyperopt iterations using local Trials
- **Stage Management**: Implemented three-tier staging (Production/Staging/Archived) with automatic performance-based assignment
- **Model Serving**: Documented multi-port serving for Production (5001), Staging (5002), and specific versions (5003)
- **Model Registry**: Tracked 4 versions with version dictionary storing metadata (version number, AUC, run_id)
- **Validation Strategy**: Added Production vs Staging comparison testing on sample data before deployment

## Results

**Model Performance**
```
Random Forest (v2)        → Production  | AUC: 0.9530
AdaBoost (v3)             → Staging     | AUC: 0.9480
Gradient Boosting (v1)    → Baseline    | AUC: 0.9664
Logistic Regression (v4)  → Archived    | AUC: 0.9252
```

**Hyperopt Best Parameters**
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.099,
    'subsample': 0.898
}
```

### MLflow UI Screenshots

![Experiment Comparison](assets/2_compare.png)
*Model comparison across all runs*

![Run Metrics](assets/3_runs.png)
*Detailed run of housing_price_classification*

![runs2](assets/4_runs-metrics.png)
*Detailed run with model metrics*

### Files Generated:

- **MLflow Tracking**: `mlruns/` directory
  - Experiment metadata
  - Run artifacts
  - Model files
- **Model Registry**: Tracked in `mlruns/models/`
- **Conda Environments**: Stored with each model
- **Visualizations**: Saved as artifacts in runs

### API Prediction Example
```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {...}}'

# Response
{"predictions": [0.005, 0.006, 0.005, 0.853, 0.009]}
```

---

**Total Runs**: 24 (1 baseline + 20 hyperopt + 3 comparison)  
**Production Model**: Random Forest v2 served on port 5001

>For detailed step-by-step instructions, see [documentation.md](documentation.md)