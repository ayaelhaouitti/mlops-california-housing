import streamlit as st
import requests

# Titre de l'application
st.title("Prédiction de Prix Immobiliers")
st.write("Entrez les caractéristiques d'une maison pour obtenir une prédiction du prix médian.")

# Formulaire pour saisir les caractéristiques
MedInc = st.number_input("Revenu médian (MedInc)", 0.0, 15.0, step=0.1)
HouseAge = st.number_input("Âge médian des maisons (HouseAge)", 1.0, 100.0, step=1.0)
AveRooms = st.number_input("Nombre moyen de pièces (AveRooms)", 1.0, 20.0, step=0.1)
AveBedrms = st.number_input("Nombre moyen de chambres (AveBedrms)", 0.5, 10.0, step=0.1)
Population = st.number_input("Population (Population)", 1.0, 50000.0, step=1.0)
AveOccup = st.number_input("Occupation moyenne (AveOccup)", 0.5, 10.0, step=0.1)
Latitude = st.number_input("Latitude", 32.0, 42.0, step=0.1)
Longitude = st.number_input("Longitude", -125.0, -114.0, step=0.1)

# Bouton pour envoyer les données à l'API
if st.button("Obtenir une prédiction"):
    # URL de l'API (ajuste selon ton déploiement)
    api_url = "http://127.0.0.1:8000/predict"
    
    # Préparation des données en format JSON
    input_data = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }
    
    # Envoi de la requête POST à l'API
    response = requests.post(api_url, json=input_data)
    
    # Vérification de la réponse
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Prix prédit : ${prediction * 100000:.2f}")
    else:
        st.error("Erreur lors de la requête à l'API.")
