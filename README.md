# Iris-Recognition-Ml
# Iris Recognition using Machine Learning

This project implements an **Iris Flower Recognition System** using **Machine Learning**.  
It predicts the species of an Iris flower based on physical measurements.

The model classifies flowers into:
- **Setosa**
- **Versicolor**
- **Virginica**

---

## Features Used
The prediction is made using these four flower features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Machine Learning Algorithm
This project uses **Support Vector Machine (SVM)** for classification.

---

## Technologies Used
- Python
- Scikit-learn
- Joblib
- Machine Learning (Classification)

---

## Dataset
The **Iris dataset** from Scikit-learn is used.  
It contains 150 samples of iris flowers with 4 features and 3 output classes.

---

##  How to Run This Project
### Step 1 – Install required libraries
Install dependencies from `Requirements.txt`:

```bash
python3 -m pip install -r Requirements.txt
```

### Step 2 – Train the model
Run the training script to generate `iris_model.pkl`:

```bash
python3 train_model.py
```

### Step 3 – Test the model
Run the test script or use the CLI to make a prediction:

```bash
python3 test_model.py
python3 predict.py --sample
```

The repository now includes a small Flask API and HTML frontend. Usage:

Start the app:

```bash
python3 app.py
```

Open http://localhost:5000 in your browser and submit feature values to get a prediction.

API endpoint:

POST `/predict` — accepts JSON or form data with keys: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.

Example JSON request:

```json
{ "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 }
```


Docker
------

Build the Docker image and run the container:

```bash
# build image
docker build -t iris-predictor:latest .

# run container (map port 5000)
docker run --rm -p 5000:5000 iris-predictor:latest
```

The app will be available at http://localhost:5000
Publish image via GitHub Actions
--------------------------------

A workflow `.github/workflows/publish-image.yml` is included to build and push the Docker image to GitHub Container Registry (GHCR) on pushes to `main` and on semver tags (e.g. `v1.2.3`).

By default the workflow pushes to `ghcr.io/<owner>/<repo>` and tags the image as `latest` and with the commit SHA. To pull the image locally:

```bash
docker pull ghcr.io/<owner>/<repo>:latest
```

If you want to publish to Docker Hub instead, set the repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` in the GitHub repository settings — the workflow will detect those secrets and push to Docker Hub as well.



### Continuous Integration
A simple GitHub Actions workflow is included at `.github/workflows/ci.yml` to run tests on push and PRs.
---

##  Purpose of This Project
This project was created to demonstrate:
- Machine Learning model training
- Classification using SVM
- Model saving and prediction


# Author
**Jagadish mattimalla**  
Computer Science & Engineering Graduate
