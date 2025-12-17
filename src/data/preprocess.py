"""
Data preprocessing and feature engineering for forest fire prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_raw_data(filepath):
    """Load raw forest fires CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath}: shape {df.shape}")
    return df


def engineer_features(df):
    """
    Feature engineering:
    - Convert month/day to numeric
    - Create binary target (fire/no-fire)
    - Normalize numeric features
    """
    df = df.copy()
    
    # Month and day mapping (handle mixed formats: numeric and string)
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    days = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

    def parse_month(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, str):
            v2 = v.strip().lower()
            return months.get(v2, pd.NA)
        try:
            return int(v)
        except Exception:
            return pd.NA

    def parse_day(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, str):
            v2 = v.strip().lower()
            # map weekday names if present, else try numeric
            if v2 in days:
                return days[v2]
            try:
                return int(v2)
            except Exception:
                return pd.NA
        try:
            return int(v)
        except Exception:
            return pd.NA

    df['month'] = df['month'].apply(parse_month).astype('Int64')
    df['day'] = df['day'].apply(parse_day).astype('Int64')
    
    # Create binary target: fire = 1 if burned area > 0, else 0
    df['fire'] = (df['area'] > 0).astype(int)
    
    # Drop original area column (we use fire for classification)
    df = df.drop(columns=['area'])

    # Cyclical encodings for month and day (handle missing by filling median)
    month_num = df['month'].astype('float').fillna(df['month'].median() if not df['month'].isna().all() else 6.0)
    day_num = df['day'].astype('float').fillna(df['day'].median() if not df['day'].isna().all() else 4.0)
    df['month_sin'] = np.sin(2 * np.pi * month_num / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_num / 12)
    df['day_sin'] = np.sin(2 * np.pi * day_num / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_num / 7)

    # Interaction features (simple, interpretable)
    if 'temp' in df.columns and 'RH' in df.columns:
        df['temp_RH'] = df['temp'] * df['RH']
    if 'temp' in df.columns and 'ISI' in df.columns:
        df['temp_ISI'] = df['temp'] * df['ISI']
    if 'wind' in df.columns and 'ISI' in df.columns:
        df['wind_ISI'] = df['wind'] * df['ISI']
    if 'FFMC' in df.columns and 'DMC' in df.columns:
        df['FFMC_DMC'] = df['FFMC'] * df['DMC']

    # Approximate BUI and FWI features (heuristic but informative)
    # BUI: combine medium- and long-term moisture (DMC + DC)
    if 'DMC' in df.columns and 'DC' in df.columns:
        df['BUI'] = df['DMC'] + df['DC']

    # FWI: combine short-term spread index (ISI) with fuel dryness (BUI)
    if 'ISI' in df.columns and 'BUI' in df.columns:
        # ensure non-negative for sqrt; add small constant to avoid zero
        buival = df['BUI'].astype(float).fillna(0.0)
        df['FWI'] = df['ISI'] * np.sqrt(np.maximum(buival, 0) + 1.0)
    
    print(f"Class distribution:\n{df['fire'].value_counts()}")
    print(f"Fire events: {df['fire'].sum()} ({100*df['fire'].mean():.1f}%)")
    
    return df


def preprocess_and_split(input_csv, output_dir='data', test_size=0.2, val_size=0.1):
    """
    Full preprocessing pipeline: load → engineer → split → save.
    
    Outputs:
    - train.csv (70%)
    - val.csv (10%)
    - test.csv (20%)
    """
    # Load and engineer
    df = load_raw_data(input_csv)
    df = engineer_features(df)
    
    # Separate features and target
    X = df.drop(columns=['fire'])
    y = df['fire']
    
    # Split: first split off test set, then split remaining into train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Calculate val_size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    # Standardize features (fit on train, apply to val/test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruct DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df['fire'] = y_train.values
    
    val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    val_df['fire'] = y_val.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_df['fire'] = y_test.values
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved train.csv: {train_df.shape}")
    print(f"Saved val.csv: {val_df.shape}")
    print(f"Saved test.csv: {test_df.shape}")
    
    return train_df, val_df, test_df, scaler


if __name__ == '__main__':
    # Example: preprocess forestfires.csv
    train_df, val_df, test_df, scaler = preprocess_and_split(
        'data/forestfires.csv',
        output_dir='data'
    )
