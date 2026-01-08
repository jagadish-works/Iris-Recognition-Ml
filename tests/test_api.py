import json


def test_get_index(client):
    res = client.get('/')
    assert res.status_code == 200


def test_predict_api(client):
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    res = client.post('/predict', data=json.dumps(sample), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert 'name' in data
    assert data['name'] == 'Setosa'
