import streamlit as st
from joblib import load
import pandas as pd

# Load the models
models = {
    "DecisionTree": load('DecisionTree.joblib'),
    "gaussianRegression": load('gaussianRegression.joblib'),
    "LinearRegression": load('LinearRegression.joblib'),
    "RandomForest": load('RandomForest.joblib'),
    "ARIMA": load('arima.joblib')
}

st.title("Groundwater Level Prediction")

# Sliders for input values
precipitation = st.slider("Precipitation", min_value=-0.000000000, max_value=0.03133407, value=0.0, step=0.0001)
evapotranspiration = st.slider("Evapotranspiration", min_value=-0.005239387, max_value=-0.041337304, value=0.0, step=0.0001)
input_data = [[precipitation, evapotranspiration]]

# Separate predict buttons for each model except ARIMA
for model_name, model in models.items():
    if model_name != "ARIMA" and st.button(f"Predict using {model_name}"):
        prediction = model.predict(input_data)[0]
        st.write(f"{model_name} Prediction: {prediction:.2f}")

# Input and predict button for ARIMA model
date_input = st.text_input("Enter a date for ARIMA prediction (dd/mm/yyyy)", "26/10/2023")
if st.button("Predict using ARIMA"):
    # Convert the date_input to the format your ARIMA model expects.
    date_for_arima = pd.to_datetime(date_input, format="%d/%m/%Y")
    prediction = models["ARIMA"].predict(date_for_arima)
    st.write(f"ARIMA Prediction: {prediction:.2f}")
