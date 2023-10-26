import streamlit as st
from joblib import load
import pandas as pd

models = {
    "DecisionTree": load('DecisionTree.joblib'),
    "LinearRegression": load('LinearRegression.joblib'),
    "RandomForest": load('RandomForest.joblib'),
    "ARIMA": load('arima.joblib')
}

st.title("Groundwater Level Prediction")

# Sliders for input values
precipitation = st.slider("Precipitation", min_value=-0.000000000, max_value=0.03133407, value=0.0, step=0.0001)
evapotranspiration = st.slider("Evapotranspiration", min_value=-0.005239387, max_value=-0.041337304, value=0.0, step=0.0001)
input_data = [[precipitation, evapotranspiration]]

for model_name, model in models.items():
    if model_name != "ARIMA" and st.button(f"Predict using {model_name}"):
        prediction = model.predict(input_data)[0]
        st.write(f"{model_name} Prediction: {prediction:.2f}")

start_date = st.date_input("Select a start date for ARIMA prediction", pd.to_datetime("26/10/2023"))
end_date = st.date_input("Select an end date for ARIMA prediction", pd.to_datetime("27/10/2023"))

if start_date > end_date:
    st.warning('End date must fall after start date.')
elif st.button("Predict using ARIMA"):
    
    prediction = models["ARIMA"].predict(start=start_date, end=end_date)
    
    st.write(f"ARIMA Predictions from {start_date} to {end_date}:")
    st.write(prediction)
