import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

models = {
    "DecisionTree": load('DecisionTree.joblib'),
    "LinearRegression": load('LinearRegression.joblib'),
    "RandomForest": load('RandomForest.joblib'),
    "ARIMA": load('arima_model.joblib')
}

st.title("Groundwater Level Prediction")

precipitation = st.slider("Precipitation", min_value=0.0, max_value=0.07312591, value=0.0, step=0.0001)
evapotranspiration = st.slider("Evapotranspiration", min_value=0.0, max_value=0.101653, value=0.0, step=0.0001)
input_data = [[precipitation, evapotranspiration]]

for model_name, model in models.items():
    if model_name != "ARIMA" and st.button(f"Predict using {model_name}"):
        prediction = model.predict(input_data)[0]
        st.write(f"{model_name} Prediction: {prediction:.2f}")

st.write("Select a date range for ARIMA prediction:")
start_date = st.date_input("Start Date", pd.to_datetime("2021-07-24"))
end_date = st.date_input("End Date", pd.to_datetime("2021-08-23"))

if start_date > end_date:
    st.warning('End date must fall after start date.')
else:
    if st.button("Predict using ARIMA"):
        try:
            last_train_date = pd.to_datetime("2021-04-18")
            forecast_steps = (pd.Timestamp(end_date) - last_train_date).days + 1 

            forecast_output = models["ARIMA"].forecast(steps=forecast_steps)
            forecast_values = forecast_output[0] if isinstance(forecast_output, (list, tuple, np.ndarray)) else [forecast_output]

            desired_forecast = forecast_values[(pd.Timestamp(start_date) - last_train_date).days:]

            forecast_df = pd.DataFrame(desired_forecast, columns=["Predicted Groundwater Level"], 
                                       index=pd.date_range(start=start_date, end=end_date))
            st.dataframe(forecast_df)
            st.line_chart(forecast_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
