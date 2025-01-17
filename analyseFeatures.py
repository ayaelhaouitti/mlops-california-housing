import shap
import matplotlib.pyplot as plt
import mlflow.sklearn  
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Charger les données d'entraînement
X_train = pd.read_csv("X_train.csv")

# Charger le modèle Random Forest depuis MLflow
model_uri = "models:/California_Housing_Best_Model/2"  
model = mlflow.sklearn.load_model(model_uri)  

# Créer un explainer SHAP pour Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Afficher les importances globales
shap.summary_plot(shap_values, X_train, plot_type="bar")
