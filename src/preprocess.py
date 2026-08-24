"""
Preprocessing for the Dengue district-month dataset.

Converts the wide-format sheet into a longformat ) and builds lag/seasonal features
suitable for a supervised forecasting model (XGBoost).
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "Dengue_Data.xlsx")
OUT_PATH = os.path.join(BASE_DIR, "data", "dengue_long_features.csv")

N_LAGS = 3  # use previous 3 months as features


def load_wide(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="WIDE DATA")
    df = df.dropna(subset=["Year"]).reset_index(drop=True)
    df = df.rename(columns={"Year": "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Two columns are both labelled "Ampara" in the source sheet; pandas
    # disambiguates the second as "Ampara.1". Drop the duplicate to avoid
    # double-counting the same district.
    if "Ampara.1" in df.columns:
        df = df.drop(columns=["Ampara.1"])

    return df


def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    district_cols = [c for c in df_wide.columns if c != "date"]
    df_long = df_wide.melt(
        id_vars="date", value_vars=district_cols,
        var_name="district", value_name="cases"
    )
    df_long = df_long.sort_values(["district", "date"]).reset_index(drop=True)

    # A few cells in the source sheet contain non-numeric placeholders
    # (e.g. "_") instead of a case count. Coerce these to NaN, then fill
    # by linear interpolation within each district's time series.
    df_long["cases"] = pd.to_numeric(df_long["cases"], errors="coerce")
    df_long["cases"] = (
        df_long.groupby("district")["cases"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df_long["cases"] = df_long["cases"].fillna(0).astype(float)
    return df_long


def add_features(df_long: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    df = df_long.copy()
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.groupby("district")["cases"].shift(lag)

    df["roll_mean_3"] = (
        df.groupby("district")["cases"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # target = next month's cases for that district
    df["target"] = df.groupby("district")["cases"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def build_dataset() -> pd.DataFrame:
    wide = load_wide()
    long_df = to_long(wide)
    featured = add_features(long_df)
    featured.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(featured)} rows, {featured['district'].nunique()} districts -> {OUT_PATH}")
    return featured


if __name__ == "__main__":
    build_dataset()
