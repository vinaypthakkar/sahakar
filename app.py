import streamlit as st
import joblib
import pandas as pd
import pmdarima as pm # Important for loading the AutoARIMA model

st.title('Demand Prediction with ARIMA Model')

# Load the pre-trained ARIMA model
model_filename = 'arima_model.joblib'
try:
    model = joblib.load(model_filename)
    st.success(f"Model '{model_filename}' loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

if model is not None:
    st.header('Make a Prediction')
    forecast_periods = st.slider(
        'Select number of periods to forecast (months):',
        min_value=1,
        max_value=24,
        value=3,
        step=1
    )

    if st.button('Generate Forecast'):
        try:
            predictions = model.predict(n_periods=forecast_periods)
            
            # Create a DataFrame for better display
            prediction_df = pd.DataFrame({
                'Date': predictions.index.strftime('%Y-%m-%d'),
                'Predicted Demand (000L)': predictions.values
            })
            st.subheader(f"Forecast for the next {forecast_periods} months:")
            st.write(prediction_df)
            
            # Optionally, plot the forecast
            st.line_chart(predictions)

        except Exception as e:
            st.error(f"Error generating forecast: {e}")
else:
    st.warning("ARIMA model could not be loaded. Please ensure 'arima_model.joblib' exists.")
