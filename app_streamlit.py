import streamlit as st
import requests
import json

st.title("🔍 Simulation d'acceptation de crédit")

# Charger les features depuis le fichier JSON utilisé par Flask
with open("mlflow_model/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

# Créer dynamiquement les champs d'entrée pour chaque feature
user_input = {}
st.subheader("🧾 Données client")
for feature in FEATURE_NAMES:
    user_input[feature] = st.number_input(f"{feature}", format="%.4f")

# Bouton pour prédire
if st.button("Prédire"):
    try:
        # Requête vers l’API Flask
        response = requests.post("http://localhost:5000/predict", json=user_input)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Probabilité de défaut : {result['probabilite_defaut']}")
            st.info(f"📋 Décision : **{result['decision']}**")
        else:
            st.error(f"Erreur : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
