# Machine Learning Predictive Pipeline

An end-to-end machine learning pipeline built with **Python** and **scikit-learn**, demonstrating the complete supervised learning workflow from raw data through to a saved, production-ready model. Uses the Titanic survival dataset to benchmark three classifiers with hyperparameter tuning and comprehensive evaluation metrics.

---

## Features

- **Automated data loading** — loads Titanic dataset via seaborn; falls back to synthetic data if unavailable
- **Feature engineering** — family size, solo traveller flag, fare-per-person derived features
- **Preprocessing pipeline** — median imputation for numerics, mode imputation + ordinal encoding for categoricals, standard scaling; all composed with `ColumnTransformer`
- **Three classifiers** — Logistic Regression, Random Forest, Gradient Boosting
- **Cross-validation** — stratified 5-fold CV to measure generalisation before tuning
- **GridSearchCV** — exhaustive hyperparameter tuning per model
- **Full evaluation metrics** — accuracy, precision, recall, F1, ROC-AUC for every model
- **Benchmark table** — side-by-side comparison across all models
- **Classification report** — per-class breakdown for the best model
- **Model persistence** — best model saved via `joblib` for reuse

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.3+ |
| Data Handling | pandas, numpy |
| Dataset | Titanic (seaborn) |
| Model Storage | joblib |
| Visualisation | matplotlib, seaborn |

---

## Pipeline Architecture

```
Raw Data
  │
  ▼
Feature Engineering
  (family_size, is_alone, fare_per_person)
  │
  ▼
ColumnTransformer
  ├── Numeric columns  → SimpleImputer(median) → StandardScaler
  └── Categorical cols → SimpleImputer(mode) → OrdinalEncoder
  │
  ▼
GridSearchCV
  ├── Logistic Regression  (C tuning)
  ├── Random Forest        (n_estimators, max_depth, min_samples_split)
  └── Gradient Boosting    (n_estimators, learning_rate, max_depth)
  │
  ▼
Evaluation
  (Accuracy, Precision, Recall, F1, ROC-AUC, CV F1)
  │
  ▼
Best Model → models/best_model.pkl
```

---

## Project Structure

```
ml-predictive-pipeline/
├── src/
│   └── pipeline.py       # Complete pipeline — load, engineer, train, tune, evaluate, save
├── models/               # Saved model files (generated on run)
├── data/                 # Place additional datasets here
└── requirements.txt
```

---

## Setup and Run

### 1. Install dependencies

```bash
git clone https://github.com/rajasuleman7/ml-predictive-pipeline.git
cd ml-predictive-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python3 src/pipeline.py
```

---

## Sample Output

```
============================================================
  MACHINE LEARNING PREDICTIVE PIPELINE
============================================================
Loaded Titanic dataset: 891 rows

Features (8): ['sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size', 'is_alone', 'fare_per_person']

Training: Logistic Regression...
  CV F1: 0.7128 ± 0.0521
  Best params: {'model__C': 1.0}

Training: Random Forest...
  CV F1: 0.7148 ± 0.0637
  Best params: {'model__max_depth': 10, 'model__min_samples_split': 5}

Training: Gradient Boosting...
  CV F1: 0.7420 ± 0.0625
  Best params: {'model__learning_rate': 0.1, 'model__max_depth': 3}

============================================================
  BENCHMARK RESULTS
============================================================
                     Accuracy  Precision  Recall      F1  ROC-AUC
Logistic Regression    0.8156     0.8103  0.6812  0.7402   0.8344
Random Forest          0.7989     0.7619  0.6957  0.7273   0.8124
Gradient Boosting      0.7989     0.7797  0.6667  0.7188   0.8098

Best model: Logistic Regression (F1 = 0.7402)
```

---

## Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')

sample = pd.DataFrame([{
    'sex': 'female', 'age': 28.0, 'sibsp': 0,
    'parch': 0, 'fare': 72.5, 'embarked': 'C',
    'family_size': 1, 'is_alone': 1, 'fare_per_person': 72.5
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]
print(f"Survived: {bool(prediction[0])}, Probability: {probability[0]:.2%}")
```

---

## Key Technical Implementations

- **`Pipeline` objects** — all preprocessing and model steps chained so train/test transforms are never leaked
- **`StratifiedKFold`** — preserves class balance in each fold for reliable CV on imbalanced classes
- **`ColumnTransformer`** — applies different preprocessing to numeric vs categorical features in a single step
- **`GridSearchCV`** — exhaustive search over param grid with CV scoring; returns the best fitted estimator
- **`joblib.dump`** — serialises the entire pipeline including fitted preprocessor for deployment-ready inference

---

## Future Improvements

- SHAP values for feature importance and model explainability
- Learning curves to diagnose bias vs variance
- Threshold tuning for precision/recall trade-off
- Additional classifiers: XGBoost, LightGBM, SVM
- Experiment tracking with MLflow
