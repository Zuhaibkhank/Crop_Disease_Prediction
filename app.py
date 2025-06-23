from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/crop_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])
    nitrogen = float(request.form['nitrogen'])
    potassium = float(request.form['potassium'])
    phosphorus = float(request.form['phosphorus'])

    data = np.array([[temp, humidity, moisture, nitrogen, potassium, phosphorus]])
    prediction = model.predict(data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
