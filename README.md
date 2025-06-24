# 🌾 Crop Disease Prediction System

A machine learning web application that predicts whether a crop is **healthy or diseased** based on soil and environmental conditions.

## 🧪 Tech Stack
- Python
- Flask
- Pandas, NumPy
- scikit-learn
- HTML & CSS (Frontend)

## 🚀 Features
- User form to input:
  - Temperature
  - Humidity
  - Moisture
  - Nitrogen, Phosphorus, Potassium
- ML model predicts crop health
- Clean and responsive UI
- Fast local prediction using trained `.pkl` model

## 📁 Project Structure


## 🧠 How It Works

1. **train_model.py**: Trains a RandomForestClassifier using crop data.
2. Saves model to `model/crop_model.pkl`
3. **app.py**: Flask web server loads the model and serves predictions via form.
4. Predictions rendered on same HTML page.

## 🔧 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model
python train_model.py

# 3. Start Flask App
python app.py
