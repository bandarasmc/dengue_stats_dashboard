"""
Trains an XGBoost regression model to forecast next-month dengue case
counts per district, using lag features + seasonal encoding + district
as a categorical feature.

Usage:
    python src/train.py
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from preprocess import build_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dengue_xgb_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "district_encoder.joblib")

FEATURE_COLS = [
    "district_enc", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "roll_mean_3",
]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.15):
    """Split by date, not randomly — the last test_frac of months (across
    all districts) become the held-out test set. This avoids leaking
    future information into training, which a random split would do."""
    dates_sorted = np.sort(df["date"].unique())
    cutoff = dates_sorted[int(len(dates_sorted) * (1 - test_frac))]
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    return train_df, test_df


def main():
    df = build_dataset()

    encoder = LabelEncoder()
    df["district_enc"] = encoder.fit_transform(df["district"])

    train_df, test_df = chronological_split(df)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # case counts can't be negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved encoder -> {ENCODER_PATH}")


if __name__ == "__main__":
    main()
