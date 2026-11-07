# STEP 1: IMPORT LIBRARIES

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 2: LOAD THE DATASET

DATA_PATH = Path("dataset/air-quality-india.csv")
MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("feature_names.joblib")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# STEP 3: PREPARE THE DATA

# Convert Timestamp into a datetime value
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Convert PM2.5 to numeric values
df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

# Remove rows with invalid timestamps or missing target values
df = df.dropna(subset=["Timestamp", "PM2.5"])

# Sort records chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Create time-based features from Timestamp
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["DayOfYear"] = df["Timestamp"].dt.dayofyear

# Cyclical features help represent repeating time patterns
df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Keep plausible non-negative PM2.5 measurements
df = df[df["PM2.5"] >= 0].copy()

print("Cleaned dataset shape:", df.shape)


# STEP 4: SELECT FEATURES AND TARGET

feature_names = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek",
    "DayOfYear",
    "HourSin",
    "HourCos",
    "MonthSin",
    "MonthCos",
]

X = df[feature_names]
y = df["PM2.5"]


# STEP 5: SPLIT DATA CHRONOLOGICALLY

# Use earlier records for training and later records for testing.
# This is more appropriate for time-based prediction than a random split.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 6: TRAIN THE MODEL

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed!")


# STEP 7: EVALUATE THE MODEL

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"MAE:  {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")
print(f"R² Score: {r2:.3f}")


# STEP 8: SAVE THE MODEL AND FEATURES

joblib.dump(model, MODEL_PATH)
joblib.dump(feature_names, FEATURES_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Feature names saved to:", FEATURES_PATH)
print("\nYou can now build the Streamlit application!")