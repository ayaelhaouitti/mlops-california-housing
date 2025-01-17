import requests

BASE_URL = "http://127.0.0.1:8000"

def test_predict_api():
    """
    Test pour vérifier que l'API de prédiction (/predict) répond correctement
    à une requête POST valide.
    """
    url = BASE_URL + "/predict"
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
    assert response.status_code == 200, "L'API /predict n'a pas renvoyé un code 200."
    
    json_response = response.json()
    assert "prediction" in json_response, "La réponse ne contient pas le champ 'prediction'."
    assert isinstance(json_response["prediction"], float), "La prédiction devrait être un float."


def test_predict_extreme_values_small():
    """
    Test de prédiction avec des valeurs très faibles (limites basses).
    """
    url = BASE_URL + "/predict"
    payload = {
        "MedInc": 0.0,
        "HouseAge": 0.0,
        "AveRooms": 0.0,
        "AveBedrms": 0.0,
        "Population": 0.0,
        "AveOccup": 0.0,
        "Latitude": 0.0,
        "Longitude": 0.0
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200, "L'API /predict n'a pas renvoyé un code 200 pour des valeurs extrêmes basses."
    
    json_response = response.json()
    assert "prediction" in json_response, "La réponse ne contient pas le champ 'prediction' pour des valeurs extrêmes basses."
    assert isinstance(json_response["prediction"], float), "La prédiction devrait être un float."


def test_predict_extreme_values_large():
    """
    Test de prédiction avec des valeurs extrêmes élevées.
    """
    url = BASE_URL + "/predict"
    payload = {
        "MedInc": 10000.0,
        "HouseAge": 100.0,
        "AveRooms": 20.0,
        "AveBedrms": 10.0,
        "Population": 1000000.0,
        "AveOccup": 50.0,
        "Latitude": 90.0,
        "Longitude": 180.0
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200, "L'API /predict n'a pas renvoyé un code 200 pour des valeurs extrêmes élevées."
    
    json_response = response.json()
    assert "prediction" in json_response, "La réponse ne contient pas le champ 'prediction' pour des valeurs extrêmes élevées."
    assert isinstance(json_response["prediction"], float), "La prédiction devrait être un float."


def test_predict_invalid_payload_missing_fields():
    """
    Test pour vérifier que l'API gère correctement un payload incomplet.
    Ici, il manque la plupart des champs attendus.
    """
    url = BASE_URL + "/predict"
    payload = {
        "MedInc": 3.0,
        "HouseAge": 30.0
    }
    response = requests.post(url, json=payload)
    assert response.status_code in (400, 422), (
        f"L'API devrait renvoyer une erreur 400/422 en cas de payload invalide. Code: {response.status_code}"
    )


def test_predict_invalid_payload_wrong_type():
    """
    Test pour vérifier que l'API gère correctement des types incorrects dans le payload.
    """
    url = BASE_URL + "/predict"
    payload = {
        "MedInc": "invalide",  # devrait être un float
        "HouseAge": 30.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    response = requests.post(url, json=payload)
    # On attend 422 ou 400
    assert response.status_code in (400, 422), (
        f"L'API devrait renvoyer une erreur 400/422 pour type de données invalide. Code: {response.status_code}"
    )


def test_predict_invalid_payload_empty():
    """
    Test pour vérifier la réaction de l'API avec un payload vide.
    """
    url = BASE_URL + "/predict"
    payload = {}
    response = requests.post(url, json=payload)
    # On attend 422 ou 400
    assert response.status_code in (400, 422), (
        f"L'API devrait renvoyer une erreur 400/422 pour un payload vide. Code: {response.status_code}"
    )
