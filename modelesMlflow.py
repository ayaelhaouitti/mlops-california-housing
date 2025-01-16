import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# **1. Charger les données**
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # Convertir en Series
y_test = pd.read_csv("y_test.csv").squeeze()  # Convertir en Series

# **2. Configurer MLflow**
mlflow.set_experiment("California_Housing_Experiment")

# **3. Fonction pour entraîner et enregistrer un modèle**
def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=f"{model_name}_Run"):
        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Calculer les métriques
        rmse = mean_squared_error(y_test, y_pred) ** 0.5  
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

        # Suivi avec MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"Modèle {model_name} enregistré avec MLflow.")

# **4. Liste des modèles**
models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42)
}

# **5. Entraîner et suivre les modèles**
for model_name, model in models.items():
    train_and_log_model(model, model_name)

print("Tous les modèles ont été entraînés et suivis dans MLflow.")
