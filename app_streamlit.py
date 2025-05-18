import streamlit as st # type: ignore
import json
import requests # type: ignore
import pandas as pd # type: ignore

# --- Configuration ---
# Assurez-vous que ce pointe vers votre fichier JSON contenant les données
DATA_FILE = 'first_5_rows.json'
# Remplacez par l'URL de votre API si elle n'est pas locale
API_URL = "https://p7-openclass.onrender.com/predict"

# --- Chargement des données ---
@st.cache_data # Cache les données pour ne pas les recharger à chaque interaction
def load_data(filepath):
    """Charge les données depuis un fichier JSON et assigne un ID basé sur l'index."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure data is a list
        if not isinstance(data, list):
             st.error(f"Erreur : Le contenu du fichier '{filepath}' n'est pas une liste JSON.")
             return None

        # Assign an 'id' based on the index (1-based) if it doesn't exist
        processed_data = []
        for index, item in enumerate(data):
            if isinstance(item, dict):
                # If 'id' key exists, use it. Otherwise, use index + 1.
                if 'id' not in item:
                    item['id'] = index + 1
                processed_data.append(item)
            else:
                st.warning(f"Ignoré un élément non-dictionnaire à l'index {index} : {item}")


        if not processed_data:
             st.error(f"Aucune donnée valide (dictionnaires avec ou sans clé 'id') trouvée dans le fichier '{filepath}'.")
             return None

        return processed_data

    except FileNotFoundError:
        st.error(f"Erreur : Le fichier de données '{filepath}' n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        st.error(f"Erreur : Impossible de décoder le fichier JSON '{filepath}'. Vérifiez sa syntaxe.")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des données : {e}")
        return None


# --- Fonction pour appeler l'API ---
def get_prediction_from_api(client_data):
    """Envoie les données client à l'API Flask et retourne la prédiction."""
    # Liste des champs requis par l'API
    required_fields = [
        "NAME_INCOME_TYPE_Working", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
        "NAME_EDUCATION_TYPE_Secondary / secondary special", "cc_PERIODE_Y_sum_sum", 
        "FLAG_EMP_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_3", "FLAG_DOCUMENT_3", 
        "CODE_GENDER", "FLAG_OWN_CAR"
    ]
    
    # Vérifiez si tous les champs requis sont présents
    missing_fields = [field for field in required_fields if field not in client_data]
    
    if missing_fields:
        error_msg = f"Les données client manquent les champs suivants requis par l'API : {', '.join(missing_fields)}"
        st.error(error_msg)
        return None
    
    try:
        response = requests.post(API_URL, json=client_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Erreur de connexion à l'API Flask. Assurez-vous que l'API est en cours d'exécution à l'adresse {API_URL}.")
        return None
    except requests.exceptions.HTTPError as e:
        # Afficher le message d'erreur renvoyé par l'API
        error_msg = "Erreur de l'API"
        try:
            error_data = e.response.json()
            error_msg = f"Erreur de l'API: {error_data}"
        except:
            error_msg = f"Erreur de l'API: {e}"
        st.error(error_msg)
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None

# --- Application Streamlit ---
st.title("Application de Prédiction de Défaut Client")

# Charger les données
all_data = load_data(DATA_FILE)

if all_data is not None:
    # Créer un dictionnaire pour un accès facile par ID
    # L'ID est maintenant garanti d'exister grâce à la fonction load_data
    data_by_id = {item['id']: item for item in all_data}

    available_ids = sorted(list(data_by_id.keys()))

    if not available_ids:
        st.error("Aucun client avec un ID valide trouvé dans les données.")
    else:
        st.sidebar.header("Sélection du Client")
        # Ensure the min/max values are based on the available IDs
        client_id = st.sidebar.number_input(
            "Entrez l'ID du client (correspond au numéro de ligne )",
            min_value=min(available_ids),
            max_value=max(available_ids),
            value=min(available_ids),
            step=1,
            format="%d"
        )

        # Valider l'ID sélectionné
        if client_id in data_by_id:
            selected_client_data = data_by_id[client_id]

            st.header(f"Données pour le Client ID: {client_id}")

            # Afficher les features individuelles de manière améliorée
            features_to_display = {k: v for k, v in selected_client_data.items() if k != 'id'}
            st.write("Variables :")

            # Utiliser st.write pour chaque variable pour une meilleure lisibilité
            for feature_name, feature_value in features_to_display.items():
                st.write(f"- **{feature_name}:** {feature_value}")


            # Bouton pour déclencher la prédiction
            if st.button("Obtenir la Prédiction"):
                st.subheader("Résultat de la Prédiction")
                # Préparer les données pour l'API (sans l'ID)
                api_input_data = features_to_display

                prediction_result = get_prediction_from_api(api_input_data)

                if prediction_result:
                    # Assurez-vous que les clés 'prediction' et 'probability' existent dans la réponse
                    prediction = prediction_result.get('prediction')
                    probability = prediction_result.get('probability')

                    if prediction is not None and probability is not None:
                        st.write(f"Prédiction : **{prediction}** (1 = Défaut, 0 = Pas de défaut)")
                        st.write(f"Probabilité de défaut : **{probability:.4f}**")

                        if prediction == 1:
                            st.error("Ce client est prédit comme étant en défaut.")
                        else:
                            st.success("Ce client est prédit comme n'étant pas en défaut.")
                    else:
                        st.warning("La réponse de l'API ne contient pas les clés 'prediction' ou 'probability'.")
                        st.json(prediction_result) # Afficher la réponse brute pour le débogage

        else:
            st.warning(f"ID client {client_id} non trouvé dans les données disponibles.")





