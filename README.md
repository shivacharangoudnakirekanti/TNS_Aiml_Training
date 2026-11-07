# Air Quality Index (AQI) Prediction Using Environmental Data

## Project Overview
This project focuses on predicting air pollution levels using historical environmental data. It uses a Machine Learning model, specifically **Random Forest Regression**, to estimate PM2.5 (Particulate Matter 2.5) concentrations based on date and time-related features.

The project also includes an interactive web application developed using Streamlit, allowing users to select a date and time and view the predicted PM2.5 concentration along with visualizations and historical data insights.

## Objectives
- Analyze historical air quality data.
- Perform data preprocessing and feature engineering.
- Develop a Machine Learning model to predict PM2.5 levels.
- Evaluate model performance using standard regression metrics.
- Build an interactive web application to display predictions.

## Technologies Used
- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning model development
- **Random Forest Regressor** – Prediction algorithm
- **Joblib** – Model saving and loading
- **Streamlit** – Interactive web application
- **Matplotlib** – Data visualization

## Dataset
The project uses a historical air quality dataset containing the following columns:

- Timestamp
- Year
- Month
- Day
- Hour
- PM2.5

The dataset is used to train and evaluate the prediction model.

## Project Workflow
1. **Data Collection:** Load the historical air quality dataset.
2. **Data Preprocessing:** Inspect and prepare the data
