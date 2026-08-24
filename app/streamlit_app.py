"""
Streamlit web app: Dengue case forecasting by district (Sri Lanka).

Loads a pre-trained XGBoost model and lets the user either:
  1) Pick a district + month to see the model's forecast against history, or
  2) Enter the last 3 months' case counts manually to get a next-month forecast.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dengue_xgb_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "district_encoder.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "dengue_long_features.csv")

st.set_page_config(page_title="Sri Lanka Dengue Forecaster", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def month_sin_cos(month: int):
    return np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)


def predict_next(model, encoder, district: str, lag1, lag2, lag3, month: int):
    district_enc = encoder.transform([district])[0]
    sin_m, cos_m = month_sin_cos(month)
    roll_mean_3 = np.mean([lag1, lag2, lag3])
    X = pd.DataFrame([{
        "district_enc": district_enc,
        "month_sin": sin_m,
        "month_cos": cos_m,
        "lag_1": lag1,
        "lag_2": lag2,
        "lag_3": lag3,
        "roll_mean_3": roll_mean_3,
    }])
    pred = model.predict(X)[0]
    return max(0.0, float(pred))


def main():
    st.title("🦟 Sri Lanka Dengue Case Forecaster")
    st.caption(
        "Forecasts next month's dengue case count for a district using an "
        "XGBoost model trained on 2010–2021 monthly surveillance data."
    )

    model, encoder = load_artifacts()
    df = load_data()
    districts = sorted(df["district"].unique())

    tab1, tab2 = st.tabs(["📈 Forecast from history", "✍️ Manual input"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            district = st.selectbox("District", districts, key="hist_district")
            dist_df = df[df["district"] == district].sort_values("date")
            last_row = dist_df.iloc[-1]
            next_month = int(last_row["date"].month % 12) + 1

            st.write(f"Using the latest available 3 months of data for **{district}**:")
            st.write(
                f"- {dist_df.iloc[-1]['date'].strftime('%b %Y')}: "
                f"{int(dist_df.iloc[-1]['cases'])} cases"
            )
            st.write(
                f"- {dist_df.iloc[-2]['date'].strftime('%b %Y')}: "
                f"{int(dist_df.iloc[-2]['cases'])} cases"
            )
            st.write(
                f"- {dist_df.iloc[-3]['date'].strftime('%b %Y')}: "
                f"{int(dist_df.iloc[-3]['cases'])} cases"
            )

            if st.button("Forecast next month", type="primary"):
                pred = predict_next(
                    model, encoder, district,
                    lag1=dist_df.iloc[-1]["cases"],
                    lag2=dist_df.iloc[-2]["cases"],
                    lag3=dist_df.iloc[-3]["cases"],
                    month=next_month,
                )
                st.success(f"Forecast: **{pred:.0f} cases** next month")

        with col2:
            st.line_chart(dist_df.set_index("date")["cases"], height=350)

    with tab2:
        st.write("Enter the last 3 months' case counts manually:")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            m_district = st.selectbox("District", districts, key="manual_district")
        with c2:
            lag1 = st.number_input("Last month", min_value=0, value=50)
        with c3:
            lag2 = st.number_input("2 months ago", min_value=0, value=50)
        with c4:
            lag3 = st.number_input("3 months ago", min_value=0, value=50)

        month = st.slider("Target month (1=Jan ... 12=Dec)", 1, 12, 6)

        if st.button("Predict", type="primary", key="manual_predict"):
            pred = predict_next(model, encoder, m_district, lag1, lag2, lag3, month)
            st.success(f"Forecast for {m_district}: **{pred:.0f} cases**")

    st.divider()
    st.caption(
        "Model: XGBoost regressor · Features: district, seasonal encoding, "
        "3-month lags, 3-month rolling mean · Trained on 2010–2021 data."
    )


if __name__ == "__main__":
    main()
