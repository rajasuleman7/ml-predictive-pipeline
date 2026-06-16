"""
Exploratory Data Analysis — Titanic Dataset
Run this before pipeline.py to understand the data.
"""

import pandas as pd
import numpy as np

def load():
    try:
        import seaborn as sns
        return sns.load_dataset('titanic')
    except:
        print("seaborn unavailable")
        return None

def explore(df):
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")

    print(f"\n--- Data Types ---")
    print(df.dtypes.to_string())

    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    pct = (missing / len(df) * 100).round(1)
    print(pd.DataFrame({'Missing': missing, 'Pct %': pct}).to_string())

    print(f"\n--- Survival Rate ---")
    print(df['survived'].value_counts(normalize=True).mul(100).round(1).to_string())

    print(f"\n--- Survival by Sex ---")
    print(df.groupby('sex')['survived'].mean().mul(100).round(1).to_string())

    print(f"\n--- Survival by Class ---")
    print(df.groupby('pclass')['survived'].mean().mul(100).round(1).to_string())

    print(f"\n--- Age Statistics ---")
    print(df['age'].describe().round(2).to_string())

    print(f"\n--- Fare Statistics ---")
    print(df['fare'].describe().round(2).to_string())

    print(f"\n--- Correlation with Survival (numeric) ---")
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr()['survived'].drop('survived').sort_values(key=abs, ascending=False)
    print(corr.round(3).to_string())

if __name__ == '__main__':
    df = load()
    if df is not None:
        explore(df)
