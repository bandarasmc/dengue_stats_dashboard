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
import plotly.express as px
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dengue_xgb_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "district_encoder.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "dengue_long_features.csv")

st.set_page_config(page_title="Sri Lanka Dengue Forecaster", layout="wide")

# Approximate district centroids (district capital / main town), matched
# EXACTLY to the district labels used in dengue_long_features.csv --
# including the dataset's non-standard names (Kalmunai, Mulativu, N Eliya).
DISTRICT_COORDS = {
    "Ampara": (7.2975, 81.6747),
    "Badulla": (6.9934, 81.0550),
    "Batticaloa": (7.7170, 81.7000),
    "Colombo": (6.9271, 79.8612),
    "Galle": (6.0535, 80.2210),
    "Gampaha": (7.0873, 80.0142),
    "Hambantota": (6.1241, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kalmunai": (7.4188, 81.8206),   # reporting unit, not an official district
    "Kalutara": (6.5854, 79.9607),
    "Kandy": (7.2906, 80.6337),
    "Kegalle": (7.2513, 80.3464),
    "Kilinochchi": (9.3961, 80.3982),
    "Kurunegala": (7.4863, 80.3647),
    "Mannar": (8.9810, 79.9044),
    "Matale": (7.4675, 80.6234),
    "Matara": (5.9549, 80.5550),
    "Moneragala": (6.8724, 81.3510),
    "Mulativu": (9.2670, 80.8142),   # Mullaitivu
    "N Eliya": (6.9497, 80.7891),    # Nuwara Eliya
    "Polonnaruwa": (7.9403, 81.0188),
    "Puttalam": (8.0362, 79.8283),
    "Ratnapura": (6.6828, 80.4012),
    "Trincomalee": (8.5874, 81.2152),
    "Vavuniya": (8.7514, 80.4971),
}
 
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


@st.cache_data
def forecast_all_districts(_model, _encoder, df, districts, target_month):
    """Run next-month forecast for every district using each district's
    latest available 3 months of cases. Returns a DataFrame with one row
    per district: district, lat, lon, latest_cases, forecast.
 
    Leading underscores on _model/_encoder tell st.cache_data not to try
    hashing those objects; df/districts/target_month are still used as the
    cache key.
    """
    rows = []
    for d in districts:
        dist_df = df[df["district"] == d].sort_values("date")
        if len(dist_df) < 3:
            continue  # not enough history to build lag features
        lag1 = dist_df.iloc[-1]["cases"]
        lag2 = dist_df.iloc[-2]["cases"]
        lag3 = dist_df.iloc[-3]["cases"]
        pred = predict_next(_model, _encoder, d, lag1, lag2, lag3, target_month)
        lat, lon = DISTRICT_COORDS.get(d, (None, None))
        rows.append({
            "district": d,
            "lat": lat,
            "lon": lon,
            "latest_cases": lag1,
            "forecast": pred,
        })
    return pd.DataFrame(rows)



def render_compare_map_tab(model, encoder, df, districts):
    st.subheader("Compare districts over time")
    default_sel = districts[:4] if len(districts) >= 4 else districts
    selected = st.multiselect(
        "Select districts to compare",
        districts,
        default=default_sel,
        max_selections=8,
    )
 
    if selected:
        comp_df = df[df["district"].isin(selected)].sort_values("date")
        fig_lines = px.line(
            comp_df,
            x="date",
            y="cases",
            color="district",
            labels={"cases": "Reported cases", "date": "Month"},
        )
        fig_lines.update_layout(height=420, legend_title_text="District")
        st.plotly_chart(fig_lines, use_container_width=True)
    else:
        st.info("Pick at least one district to see the comparison chart.")
 
    st.divider()
    st.subheader("Forecast map — all districts")
    st.caption(
        "Bubble position = approximate district centroid (not exact "
        "boundaries). Bubble size and color = forecasted next-month cases."
    )
 
    target_month = st.slider(
        "Target month for map forecast (1=Jan ... 12=Dec)",
        1, 12, 6, key="map_month",
    )
 
    with st.spinner("Forecasting all districts..."):
        map_df = forecast_all_districts(model, encoder, df, tuple(districts), target_month)
 
    if map_df.empty:
        st.warning("Not enough history to forecast any district.")
        return
 
    missing_coords = map_df[map_df["lat"].isna()]["district"].tolist()
    if missing_coords:
        st.caption(f"No coordinates for: {', '.join(missing_coords)} (skipped on map).")
    map_df = map_df.dropna(subset=["lat", "lon"])
 
    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="forecast",
        color="forecast",
        color_continuous_scale="YlOrRd",
        size_max=40,
        zoom=6.3,
        center={"lat": 7.6, "lon": 80.7},
        hover_name="district",
        hover_data={"latest_cases": True, "forecast": ":.0f", "lat": False, "lon": False},
        mapbox_style="open-street-map",
    )
    fig_map.update_layout(height=550, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
 
    st.dataframe(
        map_df[["district", "latest_cases", "forecast"]]
        .sort_values("forecast", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )
 
 
def main():
    st.title("🦟 Sri Lanka Dengue Case Forecaster")
    st.caption(
        "Forecasts next month's dengue case count for a district using an "
        "XGBoost model trained on 2010–2021 monthly surveillance data."
    )
 
    model, encoder = load_artifacts()
    df = load_data()
    districts = sorted(df["district"].unique())
 
    tab1, tab2, tab3 = st.tabs(
        ["📈 Forecast from history", "✍️ Manual input", "🗺️ Compare & Map"]
    )
 
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
 
    with tab3:
        render_compare_map_tab(model, encoder, df, districts)
 
    st.divider()
    st.caption(
        "Model: XGBoost regressor · Features: district, seasonal encoding, "
        "3-month lags, 3-month rolling mean · Trained on 2010–2021 data."
    )
 
 
if __name__ == "__main__":
    main()
 