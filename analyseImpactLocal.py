import shap
import matplotlib.pyplot as plt
import mlflow.sklearn  # Import nécessaire pour charger les modèles scikit-learn via MLflow
import pandas as pd
import numpy as np

print("Chargement des données d'entraînement...")
X_train = pd.read_csv("X_train.csv")

print("Chargement du modèle Random Forest depuis MLflow...")
model_uri = "models:/California_Housing_Best_Model/2"  # Remplace par le bon URI si nécessaire
model = mlflow.sklearn.load_model(model_uri)

print("Initialisation de SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Sélectionner plusieurs exemples aléatoires
num_examples = 3  
random_indices = np.random.choice(X_train.index, size=num_examples, replace=False)  
examples = X_train.iloc[random_indices]
examples_rounded = examples.round(2)

print(f"Exemples sélectionnés (Indices {random_indices}):")
print(examples_rounded)

# Calculer les valeurs SHAP pour les exemples sélectionnés
print("Calcul des valeurs SHAP pour les exemples sélectionnés...")
shap_values_for_examples = explainer.shap_values(examples_rounded)

for i, index in enumerate(random_indices):
    print(f"Génération du graphique SHAP pour l'exemple {i + 1} (Index {index})...")
    
    # Génération avec matplotlib pour un rendu correct
    plt.figure()  # Crée une nouvelle figure
    shap.force_plot(
        explainer.expected_value[0],  
        shap_values_for_examples[i],  
        examples_rounded.iloc[i],    
        matplotlib=True  
    )
print("Analyse locale terminée pour tous les exemples.")
