# iris_model.py
# This file trains a Machine Learning model for Iris flower recognition
# and saves the trained model for future predictions.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def train_and_save_model():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data      # Input features
    y = iris.target    # Output labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the SVM model
    model = SVC(kernel='linear')

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, "iris_model.pkl")

    print("✅ Iris model trained and saved as iris_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
