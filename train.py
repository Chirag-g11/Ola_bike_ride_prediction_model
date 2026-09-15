import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import holidays

def train_and_save_model():
    print("Starting Model Training Pipeline...")
    
    # Create artifacts directory if not exists
    os.makedirs('artifacts', exist_ok=True)

    # Note: Replace 'ola.csv' with the actual path to your dataset if running locally
    try:
        df = pd.read_csv('ola.csv')
    except FileNotFoundError:
        print("Dataset 'ola.csv' not found. Generating synthetic dataset matching Ola schema for training demonstration...")
        # Synthetic dataset fallback for pipeline execution
        np.random.seed(42)
        n_samples = 2000
        df = pd.DataFrame({
            'datetime': pd.date_range(start='2025-01-01', periods=n_samples, freq='h').astype(str),
            'season': np.random.choice([1, 2, 3, 4], size=n_samples),
            'weather': np.random.choice([1, 2, 3, 4], size=n_samples),
            'temp': np.random.uniform(10, 40, size=n_samples),
            'humidity': np.random.uniform(20, 90, size=n_samples),
            'windspeed': np.random.uniform(0, 35, size=n_samples),
            'count': np.random.randint(10, 600, size=n_samples)
        })

    # Feature Engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday.apply(lambda x: 0 if x > 4 else 1) # 1 for Weekday, 0 for Weekend

    # Holiday Calculation
    in_holidays = holidays.country_holidays('IN')
    df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: 1 if x in in_holidays else 0)

    # Feature Selection
    feature_cols = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'month', 'hour', 'weekday', 'is_holiday']
    X = df[feature_cols]
    y = df['count']

    # Train Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train XGBoost Regressor
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Save Model Artifacts
    with open('artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Success: Model and Scaler artifacts successfully created in 'artifacts/' directory!")

if __name__ == '__main__':
    train_and_save_model()