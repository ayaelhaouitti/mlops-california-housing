from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

# Initialiser l'application FastAPI
app = FastAPI(
    title="API de Prédiction de Prix Immobiliers",
    description="Cette API utilise un modèle ML pour prédire le prix médian des maisons en Californie.",
    version="1.0"
)

# Chemin relatif au modèle dans le dépôt GitHub
model_path = "mlruns/766843404691983727/217b8185706a4e4b800f2d642b2f086a/artifacts/model"

# Charger le modèle depuis MLflow
model = mlflow.pyfunc.load_model(model_path)

# Schéma pour les données d'entrée
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Point de terminaison pour les prédictions
@app.post("/predict")
def predict(input_data: HousingInput):
    # Transformer les données en tableau numpy
    features = np.array([[input_data.MedInc, input_data.HouseAge, input_data.AveRooms,
                           input_data.AveBedrms, input_data.Population, input_data.AveOccup,
                           input_data.Latitude, input_data.Longitude]])
    # Faire une prédiction
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}

# Point de terminaison pour le contrôle de santé
@app.get("/health")
def health_check():
    return {"status": "UP"}
