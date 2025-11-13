#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:27:30 2025

@author: liz
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === 1. Load dataset ===
# in the case of the BMS, this will be the dataset containg the values of
# currents, voltages, temperatures etc
data = pd.read_csv("california_housing.csv")

# === 2. Separate features (X) and target (y) ===
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# === 3. Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Train Decision Tree Regressor ===
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# === 5. Evaluate model ===
y_pred = regressor.predict(X_test)
# Calculate errors using different metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# === 6. Save model ===
joblib.dump(regressor, "regression_tree_model.pkl")
print("Model saved as 'regression_tree_model.pkl'")