import pickle
import numpy as np
from flask import Flask, render_template, request
from flask import jsonify
app = Flask(__name__)

# Load Pre-trained Artifacts
try:
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    scaler = pickle.load(open('artifacts/scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Warning: Artifacts not found. Please run 'python train.py' first.")

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """
    REST API Endpoint for programmatic inference
    """
    try:
        data = request.get_json()
        
        # Example schema validation & array formatting
        # Extract parameters from incoming JSON payload
        features = [
            float(data.get('season', 1)),
            float(data.get('weather', 1)),
            float(data.get('temp', 25.0)),
            float(data.get('humidity', 50.0)),
            float(data.get('windspeed', 10.0)),
            int(data.get('month', 1)),
            int(data.get('hour', 12)),
            int(data.get('weekday', 1)),
            int(data.get('is_holiday', 0))
        ]
        
        # Scale and Predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_demand': int(max(0, round(prediction))),
            'unit': 'rides/hour',
            'model_version': 'XGBoost-v2.4'
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract inputs from form
            season = float(request.form['season'])
            weather = float(request.form['weather'])
            temp = float(request.form['temp'])
            humidity = float(request.form['humidity'])
            windspeed = float(request.form['windspeed'])
            month = float(request.form['month'])
            hour = float(request.form['hour'])
            weekday = float(request.form['weekday'])
            is_holiday = float(request.form['is_holiday'])

            # Shape input array
            input_features = np.array([[season, weather, temp, humidity, windspeed, month, hour, weekday, is_holiday]])

            # Scale inputs & Predict
            scaled_features = scaler.transform(input_features)
            raw_prediction = model.predict(scaled_features)[0]
            
            # Format output (Demand cannot be negative)
            predicted_demand = max(0, int(round(raw_prediction)))

            return render_template(
                'index.html', 
                prediction_text=f"Estimated Ride Demand: {predicted_demand} rides/hour"
            )

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error processing input: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)