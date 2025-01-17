# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)

print("Aperçu des premières lignes :")
print(housing.data.head())

print("\nInformations sur le DataFrame :")
print(housing.data.info())

print("\nAperçu des premières valeurs cibles :")
print(housing.target.head())

print("\nRésumé statistique des variables explicatives :")
print(housing.data.describe())

print("\nVérification des valeurs manquantes :")
print(housing.data.isnull().sum())

# Définir les seuils pour les colonnes
thresholds = {
    "AveRooms": 15,      
    "AveBedrms": 6,      
    "AveOccup": 10       
}

# Compter les valeurs aberrantes
outliers_counts = {
    column: housing.data[housing.data[column] > threshold].shape[0]
    for column, threshold in thresholds.items()
}

print("\nValeurs aberrantes détectées par colonne :")
for column, count in outliers_counts.items():
    print(f"Colonne: {column}, Nombre d'aberrations détectées: {count}")

# Filtrer les données en fonction des seuils 
filtered_data = housing.data[
    (housing.data["AveRooms"] <= thresholds["AveRooms"]) &
    (housing.data["AveBedrms"] <= thresholds["AveBedrms"]) &
    (housing.data["AveOccup"] <= thresholds["AveOccup"])
]

filtered_target = housing.target.loc[filtered_data.index]

# Vérifier les tailles des données après filtrage
print(f"\nNombre de lignes dans filtered_data : {filtered_data.shape[0]}")
print(f"Nombre de lignes dans filtered_target : {filtered_target.shape[0]}")

print("\nRésumé statistique des données filtrées :")
print(filtered_data.describe())

# Diviser en train test
X_train, X_test, y_train, y_test = train_test_split(
    filtered_data, filtered_target, test_size=1/3, random_state=42
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

