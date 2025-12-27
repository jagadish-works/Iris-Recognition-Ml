# test_model.py
# This file loads the trained Iris model and predicts the flower type

import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

# Sample input: [Sepal Length, Sepal Width, Petal Length, Petal Width]
sample = [[5.1, 3.5, 1.4, 0.2]]

# Make prediction
prediction = model.predict(sample)

# Flower names
flower_names = ["Setosa", "Versicolor", "Virginica"]

print("🌸 Predicted Iris Flower Type:", flower_names[prediction[0]])
