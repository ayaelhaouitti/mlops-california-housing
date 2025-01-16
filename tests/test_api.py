import requests

# Test pour vérifier que l'API répond correctement
def test_predict_api():
    # URL de l'API locale
    url = "http://127.0.0.1:8000/predict"

    # Données d'entrée pour la prédiction
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

    # Envoyer une requête POST
    response = requests.post(url, json=payload)

    # Vérifications
    assert response.status_code == 200, "L'API n'a pas renvoyé un code 200"
    json_response = response.json()
    assert "prediction" in json_response, "La réponse ne contient pas le champ 'prediction'"
    assert isinstance(json_response["prediction"], float), "La prédiction devrait être un float"

# Test basique pour vérifier que 1 + 1 = 2 (vérifie la configuration de base)
def test_basic_math():
    assert 1 + 1 == 2, "Erreur de mathématiques basiques !"
