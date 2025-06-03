import streamlit as st # type: ignore
import json
import requests # type: ignore
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from plotly.subplots import make_subplots # type: ignore
import numpy as np

# --- Configuration ---
# Assurez-vous que ce pointe vers votre fichier JSON contenant les données
DATA_FILE = 'first_5_rows.json'
FEATURE_IMPORTANCE_FILE = 'feature_importance_global.json'
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

@st.cache_data
def load_global_feature_importance():
    """Charge l'importance globale des features."""
    try:
        with open(FEATURE_IMPORTANCE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Fichier d'importance globale '{FEATURE_IMPORTANCE_FILE}' non trouvé.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Erreur de décodage du fichier '{FEATURE_IMPORTANCE_FILE}'.")
        return {}

# --- Fonction pour créer une jauge de score ---
def create_score_gauge(probability):
    """Crée une jauge colorée pour visualiser le score de crédit."""
    # Convertir la probabilité en score (0-100)
    score = probability * 100
    
    # Définir les couleurs selon le risque
    if score < 30:
        color = "green"
        risk_level = "Faible"
    elif score < 60:
        color = "orange" 
        risk_level = "Modéré"
    else:
        color = "red"
        risk_level = "Élevé"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Score de Risque de Défaut<br><span style='font-size:0.8em;color:gray'>Niveau: {risk_level}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"}, 
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 49
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

# --- Fonction pour calculer l'importance locale des features ---
def calculate_local_feature_importance(client_data, global_importance):
    """Calcule une approximation de l'importance locale basée sur les valeurs du client."""
    local_importance = {}
    
    for feature, value in client_data.items():
        if feature in global_importance:
            # Approximation simple: importance globale * valeur normalisée
            # Plus la valeur est éloignée de 0.5, plus elle contribue
            contribution = global_importance[feature] * abs(value - 0.5) / 0.5
            local_importance[feature] = contribution
    
    return local_importance

# --- Fonction pour créer le graphique de comparaison des importances ---
def create_feature_importance_comparison(client_data, global_importance):
    """Crée un graphique comparant l'importance locale et globale des features."""
    local_importance = calculate_local_feature_importance(client_data, global_importance)
    
    # Préparer les données pour le graphique
    features = list(global_importance.keys())
    global_values = [global_importance[f] for f in features]
    local_values = [local_importance.get(f, 0) for f in features]
    client_values = [client_data.get(f, 0) for f in features]
    
    # Créer le graphique avec des sous-graphiques
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Comparaison Importance Globale vs Locale', 'Valeurs des Features pour ce Client'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Graphique 1: Comparaison des importances
    fig.add_trace(
        go.Bar(
            name='Importance Globale',
            x=features,
            y=global_values,
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Contribution Locale',
            x=features,
            y=local_values,
            marker_color='darkblue',
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Graphique 2: Valeurs des features
    colors = ['red' if v > 0.5 else 'green' for v in client_values]
    fig.add_trace(
        go.Bar(
            name='Valeurs Client',
            x=features,
            y=client_values,
            marker_color=colors,
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Mise à jour du layout
    fig.update_layout(
        height=800,
        title_text="Analyse des Features",
        showlegend=True
    )
    
    # Rotation des labels sur l'axe x
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Importance (%)", row=1, col=1)
    fig.update_yaxes(title_text="Valeur", row=2, col=1)
    
    return fig

# --- Fonction pour créer un graphique radar des features principales ---
def create_radar_chart(client_data, global_importance):
    """Crée un graphique radar des features les plus importantes."""
    # Sélectionner les 8 features les plus importantes
    sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    
    features = [f[0] for f in sorted_features]
    values = [client_data.get(f, 0) for f in features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='Profil Client',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Profil Radar du Client (Top 8 Features)",
        height=500
    )
    
    return fig

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
st.set_page_config(page_title="Prédiction de Défaut Client", layout="wide")
st.title("🏦 Application de Prédiction de Défaut Client")

# Charger les données
all_data = load_data(DATA_FILE)
global_importance = load_global_feature_importance()

if all_data is not None:
    # Créer un dictionnaire pour un accès facile par ID
    data_by_id = {item['id']: item for item in all_data}
    available_ids = sorted(list(data_by_id.keys()))

    if not available_ids:
        st.error("Aucun client avec un ID valide trouvé dans les données.")
    else:
        # Sidebar pour la sélection
        st.sidebar.header("🔍 Sélection du Client")
        client_id = st.sidebar.number_input(
            "Entrez l'ID du client",
            min_value=min(available_ids),
            max_value=max(available_ids),
            value=min(available_ids),
            step=1,
            format="%d"
        )

        # Valider l'ID sélectionné
        if client_id in data_by_id:
            selected_client_data = data_by_id[client_id]
            features_to_display = {k: v for k, v in selected_client_data.items() if k != 'id'}

            # Affichage des informations client
            st.header(f"📊 Analyse pour le Client ID: {client_id}")
            
            # Bouton pour déclencher la prédiction
            if st.button("🚀 Obtenir la Prédiction et l'Analyse", type="primary"):
                prediction_result = get_prediction_from_api(features_to_display)

                if prediction_result:
                    prediction = prediction_result.get('prediction')
                    probability = prediction_result.get('probability')

                    if prediction is not None and probability is not None:
                        # Layout en colonnes
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("🎯 Score de Risque")
                            # Jauge de score
                            gauge_fig = create_score_gauge(probability)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            # Informations textuelles
                            st.metric(
                                label="Probabilité de Défaut", 
                                value=f"{probability:.1%}",
                                delta=f"{probability - 0.5:.1%}" if probability != 0.5 else None
                            )
                            
                            if prediction == 1:
                                st.error("⚠️ Ce client est prédit comme étant en défaut.")
                            else:
                                st.success("✅ Ce client est prédit comme n'étant pas en défaut.")

                        with col2:
                            st.subheader("🎯 Profil Radar")
                            if global_importance:
                                radar_fig = create_radar_chart(features_to_display, global_importance)
                                st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Graphique de comparaison des importances (pleine largeur)
                        if global_importance:
                            st.subheader("📈 Analyse des Contributions des Variables")
                            comparison_fig = create_feature_importance_comparison(features_to_display, global_importance)
                            st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Tableau des valeurs
                            st.subheader("📋 Détail des Variables")
                            df_features = pd.DataFrame([
                                {
                                    'Variable': feature,
                                    'Valeur Client': f"{value:.3f}",
                                    'Importance Globale (%)': f"{global_importance.get(feature, 0):.1f}",
                                    'Contribution Estimée': f"{calculate_local_feature_importance(features_to_display, global_importance).get(feature, 0):.2f}"
                                }
                                for feature, value in features_to_display.items()
                            ])
                            st.dataframe(df_features, use_container_width=True)
                            
                    else:
                        st.warning("La réponse de l'API ne contient pas les clés 'prediction' ou 'probability'.")
                        st.json(prediction_result)
            
            # Affichage des données brutes (repliable)
            with st.expander("📄 Voir les données brutes du client"):
                st.json(features_to_display)

        else:
            st.warning(f"ID client {client_id} non trouvé dans les données disponibles.")