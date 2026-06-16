"""
End-to-end machine learning pipeline using scikit-learn.

Dataset: Titanic survival prediction
Target:  Binary classification — survival (1) or not (0)

Steps:
  1. Data loading
  2. Feature engineering
  3. Preprocessing (imputation, encoding, scaling)
  4. Model training with cross-validation
  5. Hyperparameter tuning (GridSearchCV)
  6. Evaluation (accuracy, precision, recall, F1, ROC-AUC)
  7. Benchmark comparison table
  8. Model persistence
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib

warnings.filterwarnings('ignore')


# ─── DATA LOADING ───────────────────────────────────────────

def load_data():
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
        print(f"Loaded Titanic dataset: {df.shape[0]} rows")
        return df
    except Exception:
        print("Generating synthetic dataset...")
        return _synthetic_data()

def _synthetic_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'survived': rng.integers(0, 2, n),
        'pclass':   rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'sex':      rng.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'age':      np.where(rng.random(n) > 0.2, rng.normal(30, 14, n).clip(1, 80), np.nan),
        'sibsp':    rng.choice(range(6), n, p=[0.68, 0.19, 0.07, 0.04, 0.01, 0.01]),
        'parch':    rng.choice(range(5), n, p=[0.76, 0.13, 0.07, 0.03, 0.01]),
        'fare':     np.where(rng.random(n) > 0.01, rng.exponential(33, n).clip(5, 520), np.nan),
        'embarked': rng.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]),
    })
    return df


# ─── FEATURE ENGINEERING ────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone']    = (df['family_size'] == 1).astype(int)
    df['fare_per_person'] = (df['fare'] / df['family_size']).round(2)
    return df


# ─── PREPROCESSING ──────────────────────────────────────────

def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
    ])


# ─── MODELS ─────────────────────────────────────────────────

def get_models():
    return {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'model__C': [0.01, 0.1, 1.0, 10.0]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators':    [100, 200],
                'model__max_depth':       [None, 5, 10],
                'model__min_samples_split': [2, 5],
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators':  [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth':     [3, 5],
            }
        },
    }


# ─── EVALUATION ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Model':        name,
        'Accuracy':     round(accuracy_score(y_test, y_pred), 4),
        'Precision':    round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':       round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1':           round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ROC-AUC':      round(roc_auc_score(y_test, y_proba), 4),
    }


# ─── MAIN ───────────────────────────────────────────────────

def run():
    print("\n" + "="*60)
    print("  MACHINE LEARNING PREDICTIVE PIPELINE")
    print("="*60)

    df = load_data()
    df = engineer_features(df)

    drop_cols = ['alive', 'who', 'adult_male', 'deck', 'embark_town',
                 'class', 'name', 'ticket', 'cabin', 'pclass']
    target    = 'survived'
    keep      = [c for c in df.columns if c != target and c not in drop_cols]
    df_clean  = df[keep + [target]].dropna(subset=[target]).copy()

    X = df_clean.drop(columns=[target])
    # Ensure string columns use object dtype (not pandas StringDtype)
    X = X.copy()
    for c in X.select_dtypes(include=["string"]).columns:
        X[c] = X[c].astype(object)
    y = df_clean[target].astype(int)

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    print(f"\nFeatures ({len(X.columns)}): {list(X.columns)}")
    print(f"Numeric:     {num_cols}")
    print(f"Categorical: {cat_cols}")
    print(f"Class balance: {dict(y.value_counts())}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    results = []
    best_model, best_score, best_name = None, 0, ''

    for name, cfg in get_models().items():
        print(f"Training: {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', cfg['model'])])

        cv = cross_val_score(pipe, X_train, y_train,
                             cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
        print(f"  CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

        gs = GridSearchCV(pipe, cfg['params'], cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"  Best params: {gs.best_params_}")

        m = evaluate(name, gs.best_estimator_, X_test, y_test)
        m['CV F1'] = round(cv.mean(), 4)
        results.append(m)

        if m['F1'] > best_score:
            best_score, best_model, best_name = m['F1'], gs.best_estimator_, name

    print("\n" + "="*60)
    print("  BENCHMARK RESULTS")
    print("="*60)
    df_res = pd.DataFrame(results).set_index('Model')
    print(df_res.to_string())

    print(f"\nBest model: {best_name} (F1 = {best_score:.4f})")
    y_pred = best_model.predict(X_test)
    print(f"\nClassification Report — {best_name}:\n")
    print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Model saved → models/best_model.pkl")
    return best_model, df_res

if __name__ == '__main__':
    run()
