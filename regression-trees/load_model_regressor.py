#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:47:17 2025

@author: liz
"""

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load model 
regressor = joblib.load("regression_tree_model.pkl")
print("Model loaded successfully!")

# Load dataset
# Here I am using the same dataset, which is not desirable, but it is just for demonstration
# In reality you will load a dataset that contains new data.
data = pd.read_csv("california_housing.csv")
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Make predictions
y_pred = regressor.predict(X)

# Evaluate and display results 
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Full dataset evaluation:")
print(f"  Mean Squared Error: {mse:.3f}")
print(f"  R² Score: {r2:.3f}")

# Show a few sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.3f}, Actual: {y.iloc[i]:.3f}")
