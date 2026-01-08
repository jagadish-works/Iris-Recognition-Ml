#!/usr/bin/env python3
import argparse
import os
import joblib
from iris_model import train_and_save_model

FLOWER_NAMES = ["Setosa", "Versicolor", "Virginica"]


def predict_sample(sample):
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        train_and_save_model()
    model = joblib.load(model_path)
    pred = model.predict([sample])[0]
    return pred


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict Iris flower species")
    parser.add_argument("--sepal-length", type=float, dest="sepal_length")
    parser.add_argument("--sepal-width", type=float, dest="sepal_width")
    parser.add_argument("--petal-length", type=float, dest="petal_length")
    parser.add_argument("--petal-width", type=float, dest="petal_width")
    parser.add_argument("--sample", action="store_true", help="Use example sample")
    args = parser.parse_args(argv)

    if args.sample or (args.sepal_length is None and args.sepal_width is None and args.petal_length is None and args.petal_width is None):
        sample = [5.1, 3.5, 1.4, 0.2]
    else:
        sample = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
        if any(v is None for v in sample):
            parser.error("Provide all four feature values or use --sample")

    pred = predict_sample(sample)
    print("🌸 Predicted Iris Flower Type:", FLOWER_NAMES[pred])
    return pred


if __name__ == "__main__":
    main()
