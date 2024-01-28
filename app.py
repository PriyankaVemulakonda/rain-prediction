from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
with open('dpmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a LabelEncoder for RainToday
label_encoder_rain_today = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    location = request.form['location']
    min_temp = float(request.form['min_temp'])
    max_temp = float(request.form['max_temp'])
    rainfall = float(request.form['rainfall'])
    wind_gust_dir = request.form['wind_gust_dir']
    wind_gust_speed = float(request.form['wind_gust_speed'])
    wind_dir_9am = request.form['wind_dir_9am']
    wind_dir_3pm = request.form['wind_dir_3pm']
    wind_speed_9am = float(request.form['wind_speed_9am'])
    wind_speed_3pm = float(request.form['wind_speed_3pm'])
    humidity_9am = float(request.form['humidity_9am'])
    humidity_3pm = float(request.form['humidity_3pm'])
    pressure_9am = float(request.form['pressure_9am'])
    pressure_3pm = float(request.form['pressure_3pm'])
    cloud_9am = float(request.form['cloud_9am'])
    cloud_3pm = float(request.form['cloud_3pm'])
    temp_9am = float(request.form['temp_9am'])
    temp_3pm = float(request.form['temp_3pm'])
    rain_today = request.form['rain_today']

    # Use the LabelEncoder to encode RainToday
    rain_today_encoded = label_encoder_rain_today.fit_transform([rain_today])

    # Prepare input features for prediction
    input_features = np.array([
        location, min_temp, max_temp, rainfall, wind_gust_dir, wind_gust_speed,
        wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm,
        humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
        cloud_9am, cloud_3pm, temp_9am, temp_3pm,
        rain_today_encoded[0]  # Use the encoded value
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)

    # Convert prediction to human-readable result
    result = 'Yes' if prediction[0] == 'yes' else 'No'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
