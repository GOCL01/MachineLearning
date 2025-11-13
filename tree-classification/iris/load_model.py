#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:04:42 2025

@author: liz
"""

import pandas as pd
import joblib

# Load the trained model and label encoder (label encoder referes to the targets)
clf = joblib.load("iris_decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Model and label encoder loaded successfully!")

# Load dataset for testing - here I am unsing the same dataset, in reality
# you need to use new data 

data = pd.read_csv("iris.csv")
X = data.drop("Species", axis=1)
y = data["Species"]

# Encode true labels for comparison
y_encoded = label_encoder.transform(y)

# Predict
y_pred_encoded = clf.predict(X)
# Here, the numerical label is transform to the actual string names
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Display first 10 predictions
for i in range(10):
    print(f"Predicted: {y_pred[i]}, Actual: {y.iloc[i]}")
