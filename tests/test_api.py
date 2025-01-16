import requests

def test_predict_api():
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "MedInc": 3.0,
        "HouseAge": 30.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
