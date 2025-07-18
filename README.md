# 🌾 Crop Disease Prediction System

A full-stack machine learning web application that predicts whether a crop is **Healthy** or affected by a specific **disease** (like Leaf Blight, Rust, Bacterial Wilt, etc.) based on soil and environmental parameters. This system aims to support farmers in identifying crop health early and taking preventive or curative actions.

---

## 🧠 Motivation

Agricultural productivity heavily depends on crop health. Manual disease diagnosis is time-consuming and error-prone. This project helps automate crop health prediction using machine learning — enabling farmers to get accurate, instant insights via a user-friendly web interface.

---

## 🚀 Features

- 🌡️ Input form to collect environmental and soil data:
  - Temperature (°C)
  - Humidity (%)
  - Moisture (%)
  - Nitrogen, Phosphorus, Potassium (NPK mg/kg)
- ⚙️ Backend ML model (RandomForestClassifier) trained on synthetic crop disease dataset.
- 📊 Real-time prediction using `.pkl` model.
- 🌐 Result opens in a **new tab** with:
  - Detected disease name (if any)
  - Symptoms and treatments
  - Prevention tips
  - Crop image based on result (Healthy or Diseased)
- 💻 Clean, animated, and responsive frontend (HTML/CSS)

---

## 🧪 Tech Stack

- **Frontend**: HTML5, CSS3 (custom animation and styling)
- **Backend**: Python, Flask
- **ML Model**: scikit-learn (`RandomForestClassifier`)
- **Libraries**: NumPy, Pandas, pickle
- **Hosting**: [Render.com](https://render.com) (Free tier)

---

## 🧠 How It Works

1. **User** enters environmental and soil parameters in a form.
2. Data is passed to a trained machine learning model.
3. Model classifies the crop as:
   - `Healthy`
   - or a type of disease like `Rust`, `Leaf Blight`, `Bacterial Wilt`, etc.
4. Results open in a new page (`/result`) with:
   - Detailed disease info
   - Treatment suggestions
   - Crop image (based on status)

---

## 📁 Project Structure
Crop_Disease_Prediction/
├── app.py # Flask server
├── train_model.py # Trains and saves the ML model
├── requirements.txt # All required Python packages
├── model/
│ └── crop_model.pkl # Trained ML model
├── dataset/
│ └── crop_disease_data.csv # Crop dataset
├── static/
│ ├── crop.jpg # Healthy crop image
│ └── diseased_crop.jpg # Diseased crop image
├── templates/
│ ├── index.html # Input form
│ └── result.html # Prediction result page
└── README.md # This file
