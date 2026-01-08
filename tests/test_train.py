import os


def test_train_creates_model(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from iris_model import train_and_save_model
        train_and_save_model()
        assert os.path.exists("iris_model.pkl")
        import joblib
        model = joblib.load("iris_model.pkl")
        assert hasattr(model, "predict")
    finally:
        os.chdir(cwd)

