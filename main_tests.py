from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import os

app = FastAPI(
    title="API de Prédiction de Prix Immobiliers",
    description="Cette API utilise un modèle ML pour prédire le prix médian des maisons en Californie.",
    version="1.0"
)

model_path = "./mlruns/196678121596541976/3e983990f0f940c9833110571636c3ba/artifacts/model"

# Charger le modèle depuis ce chemin local
model = mlflow.pyfunc.load_model(model_path)

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(input_data: HousingInput):
    features = np.array([[
        input_data.MedInc,
        input_data.HouseAge,
        input_data.AveRooms,
        input_data.AveBedrms,
        input_data.Population,
        input_data.AveOccup,
        input_data.Latitude,
        input_data.Longitude
    ]])
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}

@app.get("/health")
def health_check():
    return {"status": "UP"}
