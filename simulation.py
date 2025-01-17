import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Générer des données de production simulées
def generate_production_data(num_samples=1000):
    data = {
        "MedInc": np.random.normal(3.0, 1.0, num_samples),
        "HouseAge": np.random.randint(1, 52, num_samples),
        "AveRooms": np.random.normal(5.0, 2.0, num_samples),
        "AveBedrms": np.random.normal(1.0, 0.5, num_samples),
        "Population": np.random.randint(100, 10000, num_samples),
        "AveOccup": np.random.normal(3.0, 1.5, num_samples),
        "Latitude": np.random.uniform(32.5, 42.5, num_samples),
        "Longitude": np.random.uniform(-124.3, -114.3, num_samples)
    }
    return pd.DataFrame(data)

# Charger les données d'entraînement
training_data = pd.read_csv("X_train.csv")

# Générer des données de production simulées
production_data = generate_production_data()

# Créer un rapport Evidently pour détecter le data drift
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=training_data, current_data=production_data)

# Exporter le rapport au format HTML
data_drift_report.save_html("data_drift_report.html")
