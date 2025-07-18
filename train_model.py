import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Load dataset
data_path = 'dataset/crop_disease_data.csv'
if not os.path.exists(data_path):
    print(f"❌ Dataset not found at: {data_path}")
    exit()

df = pd.read_csv(data_path)

# Features and label
X = df[['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorus']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%\n")

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = 'model/crop_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
pickle.dump(model, open(model_path, 'wb'))
print(f"\n✅ Model saved successfully to {model_path}")
