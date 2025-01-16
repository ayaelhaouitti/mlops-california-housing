import mlflow

# ID de l'expérience dans MLflow
experiment_name = "California_Housing_Experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Récupérer les runs de l'expérience
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"]  # Trier par RMSE croissant
)

# Sélectionner le meilleur run
best_run = runs[0]  # Le premier run est celui avec le plus bas RMSE
print(f"Meilleur modèle : {best_run.info.run_id} avec RMSE = {best_run.data.metrics['rmse']}")

# Enregistrer le meilleur modèle dans le Model Registry
model_uri = f"runs:/{best_run.info.run_id}/model"
model_name = "California_Housing_Best_Model"

mlflow.register_model(model_uri, model_name)
print(f"Le meilleur modèle a été enregistré dans le Model Registry sous le nom : {model_name}")
