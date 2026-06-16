"""
Predict survival using the saved model.
Run pipeline.py first to generate models/best_model.pkl
"""

import joblib
import pandas as pd
import sys
import os

MODEL_PATH = 'models/best_model.pkl'

def predict(passenger: dict) -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run pipeline.py first.")

    model = joblib.load(MODEL_PATH)

    # Apply same feature engineering as pipeline
    data = passenger.copy()
    data['family_size']    = data.get('sibsp', 0) + data.get('parch', 0) + 1
    data['is_alone']       = int(data['family_size'] == 1)
    data['fare_per_person'] = round(data.get('fare', 0) / data['family_size'], 2)

    df = pd.DataFrame([data])
    # Cast string columns to object dtype
    for c in df.select_dtypes(include=['string']).columns:
        df[c] = df[c].astype(object)

    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        'survived':    bool(prediction),
        'probability': round(float(probability), 4),
        'confidence':  f"{probability * 100:.1f}%"
    }

# Demo passengers
EXAMPLES = [
    {'sex': 'female', 'age': 28.0, 'sibsp': 0, 'parch': 0,
     'fare': 72.5, 'embarked': 'C', 'description': '28-year-old woman, 1st class, alone'},
    {'sex': 'male',   'age': 35.0, 'sibsp': 0, 'parch': 0,
     'fare': 8.0,  'embarked': 'S', 'description': '35-year-old man, 3rd class, alone'},
    {'sex': 'female', 'age': 6.0,  'sibsp': 3, 'parch': 1,
     'fare': 31.0, 'embarked': 'S', 'description': '6-year-old girl, travelling with family'},
    {'sex': 'male',   'age': 45.0, 'sibsp': 1, 'parch': 0,
     'fare': 52.0, 'embarked': 'S', 'description': '45-year-old man, 2nd class, with spouse'},
]

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SURVIVAL PREDICTIONS — DEMO PASSENGERS")
    print("="*60 + "\n")

    for p in EXAMPLES:
        desc = p.pop('description')
        result = predict(p)
        status = "✓ SURVIVED" if result['survived'] else "✗ DID NOT SURVIVE"
        print(f"  {desc}")
        print(f"  → {status}  (confidence: {result['confidence']})\n")
