from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scalers
diabetes_model = joblib.load('diabetes_model.pkl')
diabetes_scaler = joblib.load('diabetes_scaler.pkl')

cardio_model = joblib.load('cardio_model.pkl')
cardio_scaler = joblib.load('cardio_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']

    if disease == 'diabetes':
        features = [float(request.form[x]) for x in [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'dpf', 'age'
        ]]
        scaled = diabetes_scaler.transform([features])
        result = diabetes_model.predict(scaled)[0]
        message = "Diabetic Risk Detected!" if result else "No Diabetic Risk ✅"

    elif disease == 'cardio':
        features = [float(request.form[x]) for x in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        scaled = cardio_scaler.transform([features])
        result = cardio_model.predict(scaled)[0]
        message = "Heart Disease Risk Detected!" if result else "No Heart Disease Risk ✅"

    else:
        message = "Invalid disease selected."

    return render_template('result.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)

