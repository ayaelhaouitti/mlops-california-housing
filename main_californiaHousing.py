from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

# Charger le modèle depuis MLflow
model_uri = "models:/California_Housing_Best_Model/1"  
model = mlflow.pyfunc.load_model(model_uri)

# Initialiser l'application FastAPI
app = FastAPI(
    title="API de Prédiction de Prix Immobiliers",
    description="Cette API utilise un modèle ML pour prédire le prix médian des maisons en Californie.",
    version="1.0"
)

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
