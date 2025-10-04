"""
Mini emsemble learning model with supervised learning techniques. First preprocessing
By: EssEnemiGz
Date: 4-10-2025 (DD-MM-YYYY) at 01:06 (UTC-4)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

dataset = pd.read_csv("data/cumulative_2025.10.03_21.34.32.csv", comment="#")

# Charge the dataset with the desired columns
dataset = dataset.dropna(subset=["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_impact"])
main_columns = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_impact']

x = dataset[main_columns]
y = LabelEncoder().fit_transform(dataset['koi_disposition']) # Passing strings to numeric integers

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Slicing the cases into 20% test and 80% train

# Instantiates a Random Forest classifier with 100 trees and a fixed random seed
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
joblib.dump(model, 'models/exoplanet_classifier.pkl')
