import streamlit as st
from joblib import load
import pandas as pd

# Load the models
models = {
    "DecisionTree": load('DecisionTree.joblib'),
    "LinearRegression": load('LinearRegression.joblib'),
    "RandomForest": load('RandomForest.joblib'),
    "ARIMA": load('arima_model.joblib')
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

st.write("Select a date range for ARIMA prediction:")
start_date = st.date_input("Start Date", pd.to_datetime("2021-03-04"))
end_date = st.date_input("End Date", pd.to_datetime("2021-07-24"))

if start_date > end_date:
    st.warning('End date must fall after start date.')
elif st.button("Predict using ARIMA"):
    try:
        prediction = models["ARIMA"].predict(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), typ='levels')
        st.line_chart(prediction, use_container_width=True)
    except Exception as e:
        st.error(f"Error in prediction: {e}. Ensure the provided dates are within the valid range.")
