from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model safely
model_path = os.path.join('model', 'crop_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("❌ Model file not found! Make sure 'crop_model.pkl' exists in the 'model/' folder.")

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction="Model not loaded. Please check the server logs.")

    try:
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorus = float(request.form['phosphorus'])

        data = np.array([[temp, humidity, moisture, nitrogen, potassium, phosphorus]])
        prediction = model.predict(data)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('index.html', prediction="Error processing input. Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
