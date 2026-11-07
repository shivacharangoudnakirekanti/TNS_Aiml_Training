import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime, time


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD DATA AND TRAINED MODEL
# --------------------------------------------------

DATA_PATH = Path("dataset/air-quality-india.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    feature_names = joblib.load("feature_names.joblib")
    return model, feature_names


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )
    df["PM2.5"] = pd.to_numeric(
        df["PM2.5"], errors="coerce"
    )
    df = df.dropna(subset=["Timestamp", "PM2.5"])
    return df.sort_values("Timestamp")


model, feature_names = load_model()
df = load_data()


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🌍 Air Quality Prediction System")

st.write(
    "Predict PM2.5 concentration using historical "
    "environmental data and machine learning."
)

st.info(
    "This educational application predicts PM2.5, "
    "not the official AQI from all pollutants. "
    "Results are estimates, not health advice."
)


# --------------------------------------------------
# 4. SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("Prediction Settings")

min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()

selected_date = st.sidebar.date_input(
    "Select date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

selected_time = st.sidebar.time_input(
    "Select time",
    value=time(12, 0)
)


# --------------------------------------------------
# 5. PREPARE INPUT FEATURES
# --------------------------------------------------

selected_datetime = datetime.combine(
    selected_date, selected_time
)

hour = selected_datetime.hour
month = selected_datetime.month

input_values = {
    "Year": selected_datetime.year,
    "Month": month,
    "Day": selected_datetime.day,
    "Hour": hour,
    "DayOfWeek": selected_datetime.weekday(),
    "DayOfYear": selected_datetime.timetuple().tm_yday,
    "HourSin": np.sin(2 * np.pi * hour / 24),
    "HourCos": np.cos(2 * np.pi * hour / 24),
    "MonthSin": np.sin(2 * np.pi * month / 12),
    "MonthCos": np.cos(2 * np.pi * month / 12),
}

input_df = pd.DataFrame(
    [[input_values[name] for name in feature_names]],
    columns=feature_names
)


# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

st.subheader("Selected Date and Time")

st.write(selected_datetime.strftime("%d %B %Y, %I:%M %p"))

if st.button("Predict PM2.5", type="primary"):

    prediction = float(model.predict(input_df)[0])
    prediction = max(0.0, prediction)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted PM2.5",
            value=f"{prediction:.2f} µg/m³"
        )

    # Indicative Indian PM2.5 sub-index bands
    if prediction <= 30:
        category = "Good"
        description = "PM2.5 falls in the lowest concentration band."
    elif prediction <= 60:
        category = "Satisfactory"
        description = "PM2.5 falls in the satisfactory concentration band."
    elif prediction <= 90:
        category = "Moderate"
        description = "PM2.5 falls in the moderate concentration band."
    elif prediction <= 120:
        category = "Poor"
        description = "PM2.5 falls in the poor concentration band."
    elif prediction <= 250:
        category = "Very Poor"
        description = "PM2.5 falls in the very poor concentration band."
    else:
        category = "Severe"
        description = "PM2.5 exceeds the very poor concentration band."

    with col2:
        st.metric(
            label="Indicative PM2.5 Category",
            value=category
        )

    st.write(description)

    st.caption(
        "These categories use Indian PM2.5 concentration "
        "breakpoints as an indication only. Official AQI "
        "requires pollutant sub-indices and the applicable "
        "AQI calculation method."
    )


# --------------------------------------------------
# 7. HISTORICAL DATA OVERVIEW
# --------------------------------------------------

st.divider()
st.subheader("📊 Historical PM2.5 Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric(
        "Average PM2.5",
        f"{df['PM2.5'].mean():.2f} µg/m³"
    )

with col3:
    st.metric(
        "Maximum PM2.5",
        f"{df['PM2.5'].max():.2f} µg/m³"
    )


# --------------------------------------------------
# 8. HISTORICAL TREND CHART
# --------------------------------------------------

st.subheader("PM2.5 Concentration Over Time")

# Resample to daily averages for a readable chart
daily_data = (
    df.set_index("Timestamp")["PM2.5"]
    .resample("D")
    .mean()
    .dropna()
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(daily_data.index, daily_data.values)

ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Daily Average PM2.5 Concentration")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 9. SHOW DATASET
# --------------------------------------------------

with st.expander("View sample dataset"):
    st.dataframe(df.head(100), use_container_width=True)

st.caption(
    "Machine learning project for educational purposes. "
    "Predictions are not a substitute for measured air "
    "quality or official public-health advisories."
)