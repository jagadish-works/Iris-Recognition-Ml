import os


def test_predict_cli_sample(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import predict
        # call main with sample flag
        pred = predict.main(["--sample"])
        assert pred in (0, 1, 2)
    finally:
        os.chdir(cwd)
