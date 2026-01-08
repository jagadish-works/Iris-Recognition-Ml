import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Ensure working directory is tmp_path for tests that use client
    monkeypatch.chdir(tmp_path)
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app.test_client()
