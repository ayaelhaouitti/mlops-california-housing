# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Chargement des données California Housing
housing = fetch_california_housing(as_frame=True)

# **1. Exploration des données**
# Aperçu des données
print("Aperçu des premières lignes :")
print(housing.data.head())

# Informations sur les données
print("\nInformations sur le DataFrame :")
print(housing.data.info())

# Aperçu de la variable cible
print("\nAperçu des premières valeurs cibles :")
print(housing.target.head())

# Résumé statistique des variables explicatives
print("\nRésumé statistique des variables explicatives :")
print(housing.data.describe())

# Vérification des valeurs manquantes
print("\nVérification des valeurs manquantes :")
print(housing.data.isnull().sum())

# **2. Détection des valeurs aberrantes logiques**
# Définir les seuils logiques pour les colonnes
thresholds = {
    "AveRooms": 15,      # Maximum logique pour AveRooms
    "AveBedrms": 6,      # Maximum logique pour AveBedrms
    "AveOccup": 10       # Maximum logique pour AveOccup
}

# Compter les valeurs aberrantes
outliers_counts = {
    column: housing.data[housing.data[column] > threshold].shape[0]
    for column, threshold in thresholds.items()
}

# Afficher les résultats des aberrations
print("\nValeurs aberrantes détectées par colonne :")
for column, count in outliers_counts.items():
    print(f"Colonne: {column}, Nombre d'aberrations détectées: {count}")

# **3. Filtrage des données**
# Filtrer les données en fonction des seuils logiques
filtered_data = housing.data[
    (housing.data["AveRooms"] <= thresholds["AveRooms"]) &
    (housing.data["AveBedrms"] <= thresholds["AveBedrms"]) &
    (housing.data["AveOccup"] <= thresholds["AveOccup"])
]

# Recalculer la variable cible pour qu'elle corresponde à filtered_data
filtered_target = housing.target.loc[filtered_data.index]

# Afficher les tailles des données après filtrage
print(f"\nNombre de lignes dans filtered_data : {filtered_data.shape[0]}")
print(f"Nombre de lignes dans filtered_target : {filtered_target.shape[0]}")

# Résumé statistique des données filtrées
print("\nRésumé statistique des données filtrées :")
print(filtered_data.describe())

# **4. Division des données en ensembles d'entraînement et de test**
# Diviser les données (2/3 entraînement, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(
    filtered_data, filtered_target, test_size=1/3, random_state=42
)

# **5. Sauvegarde des données divisées**
# Enregistrer les ensembles d'entraînement et de test dans des fichiers CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Indiquer que les fichiers ont été enregistrés
print("\nLes fichiers ont été enregistrés avec succès :")
print(" - 'X_train.csv'")
print(" - 'X_test.csv'")
print(" - 'y_train.csv'")
print(" - 'y_test.csv'")
