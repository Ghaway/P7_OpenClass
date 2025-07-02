import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import os
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Configuration ---
MODEL_PATH = os.environ.get('MODEL_PATH', 'file://mlflow_model')
DATA_FILE = 'first_5_rows.json'
FEATURE_IMPORTANCE_FILE = 'feature_importance_global.json'
API_URL = "https://p7-openclass.onrender.com/predict" # URL de l'API de prédiction
SHAP_API_URL = "https://p7-openclass.onrender.com/shap_values" # Nouvelle URL pour l'API SHAP


# --- Chargement des données ---
@st.cache_data
def load_data(filepath):
    """Charge les données depuis un fichier JSON et assigne un ID basé sur l'index."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
             st.error(f"Erreur : Le contenu du fichier '{filepath}' n'est pas une liste JSON.")
             return {}

        processed_data = []
        for index, item in enumerate(data):
            if isinstance(item, dict):
                if 'id' not in item:
                    item['id'] = index + 1
                processed_data.append(item)
            else:
                st.warning(f"Ignoré un élément non-dictionnaire à l'index {index} : {item}")

        if not processed_data:
             st.error(f"Aucune donnée valide trouvée dans le fichier '{filepath}'.")
             return {}

        return processed_data

    except FileNotFoundError:
        st.error(f"Erreur : Le fichier de données '{filepath}' non trouvé.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Erreur : Impossible de décoder le fichier JSON '{filepath}'. Vérifiez sa syntaxe.")
        return {}
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des données : {e}")
        return {}

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

# --- Fonction pour appeler l'API SHAP et calculer l'importance locale ---
# La fonction calculate_local_feature_importance est modifiée pour appeler l'API
def get_local_feature_importance_from_api(client_data):
    """Appelle l'API pour obtenir les valeurs SHAP et calcule l'importance locale."""
    try:
        with st.spinner('Calcul de l\'importance locale via API...'):
            response = requests.post(SHAP_API_URL, json=client_data, timeout=60) # Augmenter le timeout si nécessaire
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP

            api_result = response.json()
            
            if "shap_values" in api_result and "expected_value" in api_result:
                shap_values_raw = api_result["shap_values"]
                expected_value = api_result["expected_value"]
                
                # Convertir les valeurs SHAP en numpy array et obtenir les noms de features dans le bon ordre
                feature_names = list(shap_values_raw.keys())
                shap_values_array = np.array([shap_values_raw[feature] for feature in feature_names])
                
                # Créer le graphique en cascade (Waterfall Plot)
                st.subheader("SHAP Waterfall Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values_array,
                        base_values=expected_value,
                        feature_names=feature_names
                    ),
                    max_display=min(14, len(feature_names)),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
                # Calcul de l'importance locale normalisée (en pourcentages)
                local_importance_abs = {k: abs(v) for k, v in shap_values_raw.items()}
                total_local_importance = sum(local_importance_abs.values())
                
                if total_local_importance > 0:
                    local_importance_normalized = {k: (v/total_local_importance)*100 for k, v in local_importance_abs.items()}
                else:
                    local_importance_normalized = {k: 0 for k in local_importance_abs.keys()}
                
                return {
                    'raw': shap_values_raw,
                    'normalized': local_importance_normalized
                }
            else:
                st.error("La réponse de l'API SHAP est invalide (manque 'shap_values' ou 'expected_value').")
                st.json(api_result)
                return {'raw': {}, 'normalized': {}}

    except requests.exceptions.ConnectionError:
        st.error(f"Erreur de connexion à l'API SHAP. Assurez-vous que l'API est en cours d'exécution à l'adresse {SHAP_API_URL}.")
        return None
    except requests.exceptions.HTTPError as e:
        error_msg = f"Erreur de l'API SHAP: {e}"
        st.error(error_msg)
        st.json(response.json()) # Affiche le message d'erreur de l'API
        return None
    except requests.exceptions.Timeout:
        st.error("Timeout: L'API SHAP met trop de temps à répondre.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API SHAP : {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du traitement des valeurs SHAP : {e}")
        return None


# --- Fonction pour créer une jauge avec Plotly ---
def create_plotly_gauge(probability):
    """Crée une jauge interactive avec Plotly."""
    score = probability * 100
    threshold = 49

    # Déterminer la couleur et le niveau de risque
    if score < threshold:
        color = "green"
        risk_level = "Faible"
        risk_emoji = "🟢"
    else:
        color = "red"
        risk_level = "Élevé"
        risk_emoji = "🔴"

    # Créer la jauge avec Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{risk_emoji} Score de Risque de défaut"},
        delta = {'reference': threshold, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'lightgreen'},
                {'range': [threshold, 100], 'color': 'lightcoral'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold}
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le niveau de risque
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilité de Défaut", f"{probability:.1%}")
    with col2:
        st.metric("Seuil", "49%")
    with col3:
        st.metric("Niveau de Risque", f"{risk_emoji} {risk_level}")


# --- Fonction pour appeler l'API de prédiction ---
def get_prediction_from_api(client_data):
    """Envoie les données client à l'API Flask et retourne la prédiction."""
    required_fields = [
        "NAME_INCOME_TYPE_Working", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
        "NAME_EDUCATION_TYPE_Secondary / secondary special", "cc_PERIODE_Y_sum_sum", 
        "FLAG_EMP_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_3", "FLAG_DOCUMENT_3", 
        "CODE_GENDER", "FLAG_OWN_CAR"
    ]
    
    missing_fields = [field for field in required_fields if field not in client_data]
    
    if missing_fields:
        error_msg = f"Les données client manquent les champs suivants requis par l'API : {', '.join(missing_fields)}"
        st.error(error_msg)
        return None
    
    try:
        with st.spinner('Appel à l\'API en cours...'):
            response = requests.post(API_URL, json=client_data, timeout=30)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Erreur de connexion à l'API Flask. Assurez-vous que l'API est en cours d'exécution à l'adresse {API_URL}.")
        return None
    except requests.exceptions.HTTPError as e:
        error_msg = f"Erreur de l'API: {e}"
        st.error(error_msg)
        return None
    except requests.exceptions.Timeout:
        st.error("Timeout: L'API met trop de temps à répondre.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None

# --- Application Streamlit ---
st.set_page_config(
    page_title="Prédiction de Défaut Client", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Application de Prédiction de Défaut Client")

# Chargement modèle (n'est plus nécessaire directement pour SHAP local ici)
# Le modèle est chargé par l'API Flask maintenant
model = None # Définir à None ou supprimer si non utilisé ailleurs

# Test de connectivité API
with st.sidebar:
    st.header("🔍 Sélection du Client et Analyses")
    if st.button("🔄 Tester la connexion API de Prédiction"):
        try:
            response = requests.get(API_URL.replace('/predict', '/'), timeout=10)
            if response.status_code == 200:
                st.success("✅ API de Prédiction accessible")
            else:
                st.error(f"❌ API de Prédiction répond avec le code {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur de connexion à l'API de Prédiction: {e}")
    
    if st.button("🔄 Tester la connexion API SHAP"):
        try:
            # Envoyer un petit jeu de données de test à l'API SHAP
            # Assurez-vous que les champs correspondent aux attentes de l'API
            test_data_for_shap = {
                "NAME_INCOME_TYPE_Working": 0, "EXT_SOURCE_2": 0.5, "NAME_EDUCATION_TYPE_Higher education": 0,
                "NAME_EDUCATION_TYPE_Secondary / secondary special": 1, "cc_PERIODE_Y_sum_sum": 0,
                "FLAG_EMP_PHONE": 1, "EXT_SOURCE_1": 0.6, "EXT_SOURCE_3": 0.7, "FLAG_DOCUMENT_3": 1,
                "CODE_GENDER": 0, "FLAG_OWN_CAR": 0
            }
            response = requests.post(SHAP_API_URL, json=test_data_for_shap, timeout=10)
            if response.status_code == 200:
                st.success("✅ API SHAP accessible et répond.")
                # st.json(response.json()) # Pour voir la réponse de test
            else:
                st.error(f"❌ API SHAP répond avec le code {response.status_code}")
                st.json(response.json()) # Afficher la réponse d'erreur de l'API
        except Exception as e:
            st.error(f"❌ Erreur de connexion à l'API SHAP: {e}")


# Charger les données
all_data = load_data(DATA_FILE)
global_importance = load_global_feature_importance()

if all_data is not None:
    data_by_id = {item['id']: item for item in all_data}
    available_ids = sorted(list(data_by_id.keys()))

    if not available_ids:
        st.error("Aucun client avec un ID valide trouvé dans les données.")
    else:
        # Sidebar pour la sélection du client
        client_id = st.sidebar.selectbox(
            "Sélectionnez l'ID du client",
            options=available_ids,
            index=0
        )

        if client_id in data_by_id:
            selected_client_data = data_by_id[client_id]
            features_to_display = {k: v for k, v in selected_client_data.items() if k != 'id'}

            st.header(f"📊 Analyse pour le Client ID: {client_id}")
            
            # --- Nouvelle section dans le sidebar pour les analyses ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🚀 Options d'Analyse")

            # Checkboxes for analysis options
            selected_analyses = []
            if st.sidebar.checkbox("Scorer le client (Jauge)"):
                selected_analyses.append("score_gauge")
            if st.sidebar.checkbox("Importance locale et globale des variables"):
                selected_analyses.append("feature_importance")
            if st.sidebar.checkbox("Client dans la population (Distributions)"):
                selected_analyses.append("population_analysis")
            
            # Process selected analyses
            if selected_analyses: # Only call API if at least one analysis is selected
                prediction_result = get_prediction_from_api(features_to_display)

                if prediction_result:
                    prediction = prediction_result.get('prediction')
                    probability = prediction_result.get('probability')

                    if prediction is not None and probability is not None:
                        # 1. Scorer le client (Jauge)
                        if "score_gauge" in selected_analyses:
                            st.subheader("🎯 Score de Risque")
                            create_plotly_gauge(probability)
                            if prediction == 1:
                                st.error("⚠️ Ce client est prédit comme étant en défaut.")
                            else:
                                st.success("✅ Ce client est prédit comme n'étant pas en défaut.")

                        # 2. Importance locale et globale des variables
                        if "feature_importance" in selected_analyses:
                            if global_importance: # Plus besoin de vérifier 'model is not None' ici
                                st.subheader("📈 Analyse des Variables")
                                
                                # Appel à la nouvelle API SHAP pour l'importance locale
                                local_importance_dict = get_local_feature_importance_from_api(features_to_display)
                                
                                if local_importance_dict:
                                    local_importance_normalized = local_importance_dict['normalized']
                                    
                                    # Créer un DataFrame pour l'affichage
                                    df_analysis = pd.DataFrame([
                                        {
                                            'Variable': feature,
                                            'Valeur Client': value,
                                            'Importance Globale (%)': global_importance.get(feature, 0),
                                            'Importance Locale (%)': local_importance_normalized.get(feature, 0)
                                            
                                        }
                                        for feature, value in features_to_display.items()
                                    ]).sort_values('Importance Globale (%)', ascending=False)
                                    
                                    # Graphique avec barres comparatives (Globale vs Locale)
                                    st.subheader("📊 Comparaison Importance Globale vs Locale")
                                    
                                    # Préparer les données pour le graphique
                                    chart_data = df_analysis.set_index('Variable')[['Importance Globale (%)', 'Importance Locale (%)']]
                                    
                                    # Créer le graphique avec Plotly pour plus de contrôle
                                    fig = go.Figure()
                                    
                                    # Ajouter les barres pour l'importance globale
                                    fig.add_trace(go.Bar(
                                        name='Importance Globale (%)',
                                        x=chart_data.index,
                                        y=chart_data['Importance Globale (%)'],
                                        marker_color='lightblue',
                                        opacity=0.8
                                    ))
                                    
                                    # Ajouter les barres pour l'importance locale
                                    fig.add_trace(go.Bar(
                                        name='Importance Locale (%)',
                                        x=chart_data.index,
                                        y=chart_data['Importance Locale (%)'],
                                        marker_color='orange',
                                        opacity=0.8
                                    ))
                                    
                                    # Mise à jour du layout
                                    fig.update_layout(
                                        title="Comparaison des Importances des Variables",
                                        xaxis_title="Variables",
                                        yaxis_title="Importance (%)",
                                        barmode='group',
                                        height=500,
                                        xaxis_tickangle=-45
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Tableau détaillé
                                    st.subheader("📋 Tableau Détaillé des Variables")
                                    st.dataframe(df_analysis, use_container_width=True)
                                else:
                                    st.warning("⚠️ Impossible de récupérer l'importance locale des variables depuis l'API SHAP.")

                            else:
                                st.warning("⚠️ Fichier d'importance globale non disponible")
                        
                        # 3. Client dans la population (Distributions)
                        if "population_analysis" in selected_analyses:
                            st.subheader("📦 Analyse du Client dans la Population")

                            boxplot_json_path = "boxplot_stats.json"
                            bool_stats_path = "bool_stats.json"

                            try:
                                # Charger les statistiques
                                with open(boxplot_json_path, "r") as f:
                                    boxplot_stats = json.load(f)
                                with open(bool_stats_path, "r", encoding="utf-8") as f:
                                    bool_stats = json.load(f)

                                # Déterminer les variables continues et booléennes disponibles
                                # On filtre sur les clés présentes dans features_to_display ET dans les fichiers de stats
                                continuous_vars = [
                                    var for var in features_to_display.keys() 
                                    if var in boxplot_stats.get('0', {}) and var in boxplot_stats.get('1', {})
                                ]
                                bool_vars = [
                                    var for var in features_to_display.keys() 
                                    if features_to_display[var] in [0, 1] and 
                                    var in bool_stats.get('0', {}) and var in bool_stats.get('1', {})
                                ]
                                
                                all_dist_vars = sorted(list(set(continuous_vars + bool_vars)))

                                if not all_dist_vars:
                                    st.warning("Aucune variable continue ou booléenne pertinente trouvée pour l'analyse des distributions.")
                                else:
                                    selected_dist_var = st.selectbox(
                                        "Sélectionnez une variable à analyser :",
                                        options=all_dist_vars
                                    )

                                    if selected_dist_var in continuous_vars:
                                        st.subheader(f"Boxplot pour '{selected_dist_var}'")
                                        # Créer une colonne pour contenir le graphique et réduire sa largeur
                                        col_graph, _ = st.columns([0.6, 0.4]) # 60% de largeur pour le graph, 40% vide
                                        with col_graph:
                                            fig, ax = plt.subplots(figsize=(6, 4)) 

                                            box_data = [
                                                {
                                                    'med': boxplot_stats['0'][selected_dist_var]['median'],
                                                    'q1': boxplot_stats['0'][selected_dist_var]['q1'],
                                                    'q3': boxplot_stats['0'][selected_dist_var]['q3'],
                                                    'whislo': boxplot_stats['0'][selected_dist_var]['min'],
                                                    'whishi': boxplot_stats['0'][selected_dist_var]['max'],
                                                    'fliers': [],
                                                    'label': 'Target 0'
                                                },
                                                {
                                                    'med': boxplot_stats['1'][selected_dist_var]['median'],
                                                    'q1': boxplot_stats['1'][selected_dist_var]['q1'],
                                                    'q3': boxplot_stats['1'][selected_dist_var]['q3'],
                                                    'whislo': boxplot_stats['1'][selected_dist_var]['min'],
                                                    'whishi': boxplot_stats['1'][selected_dist_var]['max'],
                                                    'fliers': [],
                                                    'label': 'Target 1'
                                                }
                                            ]

                                            ax.bxp(box_data, showfliers=False)
                                            ax.set_title(selected_dist_var, fontsize=12)
                                            ax.tick_params(axis='both', labelsize=10)
                                            client_val = features_to_display.get(selected_dist_var)
                                            if client_val is not None:
                                                ax.axhline(client_val, color='red', linestyle='--', label=f'Client: {client_val:.2f}')
                                                ax.legend(fontsize=10)

                                            st.pyplot(fig, use_container_width=True) # Garder use_container_width=True pour qu'il remplisse la petite colonne
                                            plt.close(fig) # Important pour éviter les avertissements Streamlit

                                    elif selected_dist_var in bool_vars:
                                        st.subheader(f"Distribution pour '{selected_dist_var}'")
                                        target0 = bool_stats['0'][selected_dist_var]
                                        target1 = bool_stats['1'][selected_dist_var]

                                        df_plot = pd.DataFrame({
                                            "Valeur": [0, 1],
                                            "Target 0 (%)": [target0["0"], target0["1"]],
                                            "Target 1 (%)": [target1["0"], target1["1"]]
                                        })

                                        # Créer une colonne pour contenir le graphique et réduire sa largeur
                                        col_graph, _ = st.columns([0.6, 0.4]) # 60% de largeur pour le graph, 40% vide
                                        with col_graph:
                                            fig, ax = plt.subplots(figsize=(6, 4))
                                            width = 0.35
                                            x = np.arange(len(df_plot["Valeur"]))

                                            bars1 = ax.bar(x - width/2, df_plot["Target 0 (%)"], width=width, color="green", label="Target 0")
                                            bars2 = ax.bar(x + width/2, df_plot["Target 1 (%)"], width=width, color="red", label="Target 1")

                                            ax.set_xticks(x)
                                            ax.set_xticklabels(["0", "1"])
                                            ax.set_ylim(0, 100)
                                            ax.set_ylabel("Pourcentage")
                                            ax.set_title(selected_dist_var, fontsize=12)
                                            
                                            client_val = features_to_display.get(selected_dist_var)
                                            if isinstance(client_val, (int, float)) and int(client_val) in [0, 1]:
                                                client_val_int = int(client_val)
                                                if client_val_int == 0:
                                                    bars1[0].set_edgecolor('black')
                                                    bars1[0].set_linewidth(3)
                                                    bars2[0].set_edgecolor('black') 
                                                    bars2[0].set_linewidth(3)
                                                else:
                                                    bars1[1].set_edgecolor('black')
                                                    bars1[1].set_linewidth(3)
                                                    bars2[1].set_edgecolor('black')
                                                    bars2[1].set_linewidth(3)
                                                
                                                from matplotlib.patches import Rectangle
                                                legend_elements = [
                                                    Rectangle((0, 0), 1, 1, facecolor='green', label='Target 0'),
                                                    Rectangle((0, 0), 1, 1, facecolor='red', label='Target 1'),
                                                    Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=3, label=f'Client: {client_val_int}')
                                                ]
                                                ax.legend(handles=legend_elements, fontsize=10)
                                            else:
                                                ax.legend(fontsize=10)

                                            st.pyplot(fig, use_container_width=True) # Garder use_container_width=True
                                            plt.close(fig) # Important
                                        
                            except FileNotFoundError:
                                st.warning("📁 Fichier 'boxplot_stats.json' ou 'bool_stats.json' introuvable. Merci de les générer avec les stats.")
                            except Exception as e:
                                st.error(f"Erreur lors de l'affichage des distributions : {e}")

                    else:
                        st.warning("La réponse de l'API ne contient pas les clés 'prediction' ou 'probability'.")
                        st.json(prediction_result)
                else:
                    st.warning("Impossible d'obtenir la prédiction de l'API. Veuillez vérifier la connexion.")
            else: # Si aucune analyse n'est sélectionnée
                st.info("Veuillez sélectionner une ou plusieurs options d'analyse dans la barre latérale pour voir les résultats.")

            # Affichage des données brutes
            with st.expander("📄 Voir les données brutes du client"):
                st.json(features_to_display)

        else:
            st.warning(f"ID client {client_id} non trouvé dans les données disponibles.")

# Footer
st.markdown("---")
st.markdown("💡 **Astuce**: Si vous rencontrez des problèmes, vérifiez que l'API est bien déployée et accessible.")