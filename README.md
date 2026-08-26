# Sri Lanka Dengue Case Forecaster

An end-to-end AI web application that forecasts next-month dengue case
counts for districts in Sri Lanka, using historical monthly surveillance
data (2010–2021).

## 1. Problem Statement

Dengue fever is a recurring public health burden in Sri Lanka, with case
counts varying sharply by district and season. Health authorities and
resource planners need early estimates of expected case volumes to plan
vector-control campaigns, hospital resource allocation, and public
awareness efforts. This project addresses the problem of **forecasting
next-month dengue case counts at the district level** using historical
surveillance data.

## 2. Use Case

The application can be used by:
- Public health planners to get a quick, data-driven estimate of next
  month's dengue caseload for a given district before allocating
  resources.
- Researchers/students exploring time-series forecasting on
  epidemiological data.
- Anyone who wants to enter recent case counts for a district and see a
  short-term forecast.

## 3. Solution Overview

The app trains a supervised regression model on 12 years of monthly,
district-level dengue case counts. For each district-month, it uses the
previous three months' case counts, a 3-month rolling average, and a
seasonal (month) encoding to predict the following month's case count.
Users interact with the model through a Streamlit web interface: either
by selecting a district and letting the app pull its most recent history,
or by manually entering the last three months' case counts for a
hypothetical scenario.

## 4. Dataset

- **Source**: `data/Dengue_Data.xlsx` (provided monthly dengue
  surveillance counts for Sri Lanka).
- **Coverage**: January 2010 – December 2021 (144 months), 25 districts.
- **Format**: Wide format in the source file (one row per month, one
  column per district); reshaped by `src/preprocess.py` into a long
  format (one row per district-month) with derived features.
- **Data cleaning notes**:
  - One duplicate "Ampara" column in the source sheet is dropped.
  - A small number of non-numeric placeholder values (e.g. `"_"`) are
    treated as missing and filled via linear interpolation within each
    district's time series.

## 5. AI/ML Approach

- **Model**: `XGBRegressor` (XGBoost), a gradient-boosted tree regressor.
- **Why**: With ~3,500 district-month training rows, XGBoost gives
  strong performance without the data volume that deep sequence models
  (LSTM/GNN) typically need, and it's fast to train, interpret, and
  deploy — appropriate for this assignment's scope.
- **Features**:
  - `district` (label-encoded)
  - `month_sin`, `month_cos` (cyclical encoding of seasonality)
  - `lag_1`, `lag_2`, `lag_3` (previous 3 months' case counts)
  - `roll_mean_3` (3-month rolling average)
- **Target**: next month's case count for that district.
- **Train/test split**: chronological (last ~15% of months held out),
  not random — this avoids leaking future information into training.
- **Evaluation**: MAE and RMSE on the held-out period (see
  `src/train.py` output).
- **Libraries**: pandas, numpy, scikit-learn, xgboost, joblib.

## 6. Application Architecture

```
Dengue_Data.xlsx
      │
      ▼
src/preprocess.py   (reshape wide→long, clean, feature engineering)
      │
      ▼
src/train.py         (train XGBoost model, save model + encoder)
      │
      ▼
models/*.joblib       (trained model artifacts)
      │
      ▼
app/streamlit_app.py  (loads model, serves interactive web UI)
```

The Streamlit app loads the pre-trained model directly (no separate API
layer) and serves both the UI and inference logic in one process —
appropriate for this assignment's single-container deployment.

## 7. Technology Stack

- **Language**: Python 3.12
- **ML**: XGBoost, scikit-learn
- **Data**: pandas, numpy, openpyxl
- **Web UI**: Streamlit
- **Containerization**: Docker

## 8. Local Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/bandarasmc/dengue_stats_dashboard.git
cd dengue_stats_dashboard

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprocess data and train the model
#    (skip this step if models/*.joblib are already included in the repo)
cd src
python preprocess.py
python train.py
cd ..

# 5. Run the app
python -m streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`.

## 9. Deployment Details

This project is **containerized with Docker** and ready for deployment
to any container-hosting cloud service (e.g. AWS ECS/App Runner, Azure
Container Apps, Google Cloud Run) but has not been deployed to a live
cloud environment for this submission. The Docker image can be built and
run locally, and pushed to a container registry, as shown below.

## 10. Web Application Usage

1. Open the app in your browser (`http://localhost:8501` locally).
2. **Forecast from history tab**: pick a district; the app shows its
   most recent 3 months of recorded cases and a historical chart. Click
   "Forecast next month" to get a prediction.
3. **Manual input tab**: pick a district, enter any 3 months of case
   counts and a target month, and click "Predict" to get a forecast for
   a custom scenario.

## 11. Docker Instructions
1. Log in to Docker Hub:

```bash
docker login
```

2. Build the image:
```bash
docker build -f Dockerfile -t streamlit-webapp:latest .
```

3. Push the Streamlit image to Docker Hub:

```bash
docker tag streamlit-webapp:latest vilochana94/streamlit-webapp:latest
docker push vilochana94/streamlit-webapp:latest
```


4. Run the Streamlit container:

```bash
docker run --rm -p 8501:8501 streamlit-webapp:latest
```


Then open `http://localhost:8501` in your browser.



## Project Structure

```
dengue-forecast/
├── app/
│   └── streamlit_app.py       # Streamlit web application
├── src/
│   ├── preprocess.py          # Data cleaning & feature engineering
│   └── train.py                # Model training
├── data/
│   ├── Dengue_Data.xlsx        # Raw dataset
│   └── dengue_long_features.csv  # Generated by preprocess.py
├── models/
│   ├── dengue_xgb_model.joblib   # Trained XGBoost model
│   └── district_encoder.joblib   # District label encoder
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```
