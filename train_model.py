import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('dataset/crop_disease_data.csv')

X = df[['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorus']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('model/crop_model.pkl', 'wb'))
print("Model trained and saved as crop_model.pkl")

