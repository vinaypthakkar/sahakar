from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pmdarima as pm # Required for loading AutoARIMA model

app = Flask(__name__)

# Load the pre-trained ARIMA model
model_filename = 'arima_model.joblib'
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "ARIMA Model Prediction API. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.get_json(force=True)
    
    # The ARIMA model predicts 'n_periods' into the future.
    # For this example, let's assume the client sends the number of periods to forecast.
    try:
        forecast_periods = data.get('forecast_periods', 1) # Default to 1 period if not specified
        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            return jsonify({'error': 'forecast_periods must be a positive integer.'}), 400

        predictions = model.predict(n_periods=forecast_periods)
        
        # Convert predictions to a list or dictionary for JSON serialization
        # Assuming predictions is a pandas Series with a datetime index
        prediction_output = {
            'index': predictions.index.strftime('%Y-%m-%d').tolist(),
            'predictions': predictions.tolist()
        }
        
        return jsonify(prediction_output)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # To run the app, make sure arima_model.joblib is in the same directory
    # or provide the full path.
    # You can run this from your terminal using: python app.py
    # For production, use a production-ready WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000)
