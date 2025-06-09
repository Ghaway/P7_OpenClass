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
API_URL = "https://p7-openclass.onrender.com/predict"

# --- Chargement des données ---
@st.cache_data
def load_data(filepath):
    """Charge les données depuis un fichier JSON et assigne un ID basé sur l'index."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
             st.error(f"Erreur : Le contenu du fichier '{filepath}' n'est pas une liste JSON.")
             return None

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

# --- Fonction pour calculer l'importance locale ---
def calculate_local_feature_importance(model, client_data):
    """Calcule l'importance locale basée sur les valeurs du client."""
    try:
        # Vérifier si le modèle est un Pipeline et extraire le modèle de base
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # Obtenir le dernier step du pipeline (généralement le modèle)
            base_model = model.steps[-1][1]
            st.info(f"Pipeline détecté, utilisation du modèle: {type(base_model).__name__}")
        else:
            base_model = model
        
        # Préparer les données
        client_data_array = np.array([list(client_data.values())])
        feature_names = list(client_data.keys())
        
        # Si c'est un pipeline, on doit transformer les données avec les étapes de preprocessing
        if isinstance(model, Pipeline):
            # Appliquer toutes les transformations sauf la dernière étape (le modèle)
            transformed_data = client_data_array
            for step_name, transformer in model.steps[:-1]:
                if hasattr(transformer, 'transform'):
                    transformed_data = transformer.transform(transformed_data)
            
            # Utiliser les données transformées pour SHAP
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(transformed_data)
        else:
            # Si ce n'est pas un pipeline, utiliser directement
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(client_data_array)

        # Afficher les informations de débogage
        st.info(f"Forme des valeurs SHAP: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
        
        # Gérer le cas où shap_values peut être une liste (classification binaire)
        if isinstance(shap_values, list):
            # Pour la classification binaire, prendre les valeurs de la classe positive (index 1)
            if len(shap_values) == 2:
                shap_values_to_plot = shap_values[1][0]  # Première observation, classe positive
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:
                shap_values_to_plot = shap_values[0][0]
                expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        else:
            # Cas où shap_values est un array numpy
            if len(shap_values.shape) == 3:  # (n_samples, n_features, n_classes)
                shap_values_to_plot = shap_values[0, :, 1]  # Première observation, classe positive
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            elif len(shap_values.shape) == 2:  # (n_samples, n_features)
                shap_values_to_plot = shap_values[0, :]  # Première observation
                expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:  # Array 1D
                shap_values_to_plot = shap_values
                expected_value = explainer.expected_value

        # Créer le graphique en cascade
        st.subheader("SHAP Waterfall Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # S'assurer que la longueur des features correspond
        min_len = min(len(feature_names), len(shap_values_to_plot))
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_to_plot[:min_len],
                base_values=expected_value,
                feature_names=feature_names[:min_len]
            ),
            max_display=min(14, min_len),
            show=False
        )
        st.pyplot(fig)
        plt.close()
        
        # Retourner un dictionnaire avec les valeurs SHAP pour chaque feature
        return dict(zip(feature_names[:min_len], shap_values_to_plot[:min_len]))
        
    except Exception as e:
        st.error(f"Erreur lors du calcul de l'importance locale: {e}")
        st.error(f"Type de modèle: {type(model)}")
        
        # Essayer une approche alternative avec un explainer générique
        try:
            st.info("Tentative avec un explainer générique...")
            # Créer des données de background simples
            background_data = np.zeros((1, len(client_data)))
            explainer = shap.Explainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, 
                                     background_data)
            shap_values = explainer(np.array([list(client_data.values())]))
            
            # Extraire les valeurs pour la classe positive si c'est de la classification
            if len(shap_values.values.shape) > 2:
                values = shap_values.values[0, :, 1]  # classe positive
            else:
                values = shap_values.values[0]
                
            return dict(zip(list(client_data.keys()), values))
            
        except Exception as e2:
            st.error(f"Erreur avec l'explainer générique: {e2}")
            return {}

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
        title = {'text': f"{risk_emoji} Score de Risque de Défaut"},
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


# --- Fonction pour appeler l'API ---
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
        error_msg = "Erreur de l'API"
        try:
            error_data = e.response.json()
            error_msg = f"Erreur de l'API: {error_data}"
        except:
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

# Chargement modèle
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    st.success("✅ Modèle chargé avec succès")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle: {e}")
    model = None

# Test de connectivité API
with st.sidebar:
    st.header("🔍 Sélection du Client")
    if st.button("🔄 Tester la connexion API"):
        try:
            response = requests.get(API_URL.replace('/predict', '/'), timeout=10)
            if response.status_code == 200:
                st.success("✅ API accessible")
            else:
                st.error(f"❌ API répond avec le code {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur de connexion: {e}")

# Charger les données
all_data = load_data(DATA_FILE)
global_importance = load_global_feature_importance()

if all_data is not None:
    data_by_id = {item['id']: item for item in all_data}
    available_ids = sorted(list(data_by_id.keys()))

    if not available_ids:
        st.error("Aucun client avec un ID valide trouvé dans les données.")
    else:
        # Sidebar pour la sélection
        client_id = st.sidebar.selectbox(
            "Sélectionnez l'ID du client",
            options=available_ids,
            index=0
        )

        if client_id in data_by_id:
            selected_client_data = data_by_id[client_id]
            features_to_display = {k: v for k, v in selected_client_data.items() if k != 'id'}

            st.header(f"📊 Analyse pour le Client ID: {client_id}")
            
            # Bouton pour déclencher la prédiction
            if st.button("🚀 Obtenir la Prédiction et l'Analyse", type="primary"):
                prediction_result = get_prediction_from_api(features_to_display)

                if prediction_result:
                    prediction = prediction_result.get('prediction')
                    probability = prediction_result.get('probability')

                    if prediction is not None and probability is not None:
                        
                        # Jauge de score avec Plotly
                        st.subheader("🎯 Score de Risque")
                        create_plotly_gauge(probability)
                        
                        # Status
                        if prediction == 1:
                            st.error("⚠️ Ce client est prédit comme étant en défaut.")
                        else:
                            st.success("✅ Ce client est prédit comme n'étant pas en défaut.")
                        
                        # Analyse des features si disponible
                        if global_importance and model is not None:
                            st.subheader("📈 Analyse des Variables")
                            
                            # Fix: Use features_to_display instead of undefined client_data
                            local_importance = calculate_local_feature_importance(model, features_to_display)
                            
                            # Créer un DataFrame pour l'affichage
                            df_analysis = pd.DataFrame([
                                {
                                    'Variable': feature,
                                    'Valeur Client': value,
                                    'Importance Globale (%)': global_importance.get(feature, 0),
                                    'Contribution Locale (SHAP)': local_importance.get(feature, 0)
                                }
                                for feature, value in features_to_display.items()
                            ]).sort_values('Importance Globale (%)', ascending=False)
                            
                            # Graphique simple avec Streamlit
                            st.bar_chart(df_analysis.set_index('Variable')['Importance Globale (%)'])
                            
                            # Tableau détaillé
                            st.dataframe(df_analysis, use_container_width=True)

                            # --- Affichage Boxplots pour variables continues ---
                            st.subheader("📦 Analyse par Boxplots (comparaison client vs population)")

                            boxplot_json_path = "boxplot_stats.json"  # <- adapter si nécessaire
                            continuous_vars = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "cc_PERIODE_Y_sum_sum"]  # <- adapte à tes vraies variables continues

                            try:
                                with open(boxplot_json_path, "r") as f:
                                    boxplot_stats = json.load(f)

                                from itertools import zip_longest

                                # Groupes de 2 variables par ligne (2x2)
                                grouped_vars = list(zip_longest(*(iter(continuous_vars),) * 2))

                                for row_vars in grouped_vars:
                                    cols = st.columns(2)
                                    for i, var in enumerate(row_vars):
                                        if var is None:
                                            continue

                                        with cols[i]:
                                            if var not in boxplot_stats['0'] or var not in boxplot_stats['1']:
                                                st.warning(f"{var} non trouvée dans le fichier de stats.")
                                                continue

                                            fig, ax = plt.subplots(figsize=(4, 3))  # Réduction taille

                                            box_data = [
                                                {
                                                    'med': boxplot_stats['0'][var]['median'],
                                                    'q1': boxplot_stats['0'][var]['q1'],
                                                    'q3': boxplot_stats['0'][var]['q3'],
                                                    'whislo': boxplot_stats['0'][var]['min'],
                                                    'whishi': boxplot_stats['0'][var]['max'],
                                                    'fliers': [],
                                                    'label': 'Target 0'
                                                },
                                                {
                                                    'med': boxplot_stats['1'][var]['median'],
                                                    'q1': boxplot_stats['1'][var]['q1'],
                                                    'q3': boxplot_stats['1'][var]['q3'],
                                                    'whislo': boxplot_stats['1'][var]['min'],
                                                    'whishi': boxplot_stats['1'][var]['max'],
                                                    'fliers': [],
                                                    'label': 'Target 1'
                                                }
                                            ]

                                            ax.bxp(box_data, showfliers=False)
                                            ax.set_title(var, fontsize=10)
                                            ax.tick_params(axis='both', labelsize=8)
                                            client_val = features_to_display.get(var)
                                            if client_val is not None:
                                                ax.axhline(client_val, color='red', linestyle='--', label='Client')
                                                ax.legend(fontsize=8)

                                            st.pyplot(fig)

                            except FileNotFoundError:
                                st.warning("📁 Fichier 'boxplot_stats.json' introuvable. Merci de le générer avec les stats.")
                            except Exception as e:
                                st.error(f"Erreur lors de l'affichage des boxplots : {e}")
                            
                            # --- Affichage des distributions booléennes CORRIGÉ ---
                            st.subheader("📊 Analyse des Variables Booléennes (par classe cible)")

                            bool_stats_path = "bool_stats.json"
                            bool_vars = [var for var in features_to_display.keys() if features_to_display[var] in [0, 1] and var not in continuous_vars]

                            try:
                                with open(bool_stats_path, "r", encoding="utf-8") as f:
                                    bool_stats = json.load(f)

                                # Affichage en grille 2x2
                                from itertools import zip_longest
                                grouped_bools = list(zip_longest(*(iter(bool_vars),) * 2))

                                for row_vars in grouped_bools:
                                    cols = st.columns(2)
                                    for i, var in enumerate(row_vars):
                                        if var is None or var not in bool_stats['0'] or var not in bool_stats['1']:
                                            continue

                                        with cols[i]:
                                            target0 = bool_stats['0'][var]
                                            target1 = bool_stats['1'][var]

                                            df_plot = pd.DataFrame({
                                                "Valeur": [0, 1],
                                                "Target 0 (%)": [target0["0"], target0["1"]],
                                                "Target 1 (%)": [target1["0"], target1["1"]]
                                            })

                                            fig, ax = plt.subplots(figsize=(4, 3))
                                            width = 0.35
                                            x = np.arange(len(df_plot["Valeur"]))

                                            # Barres
                                            bars1 = ax.bar(x - width/2, df_plot["Target 0 (%)"], width=width, color="green", label="Target 0")
                                            bars2 = ax.bar(x + width/2, df_plot["Target 1 (%)"], width=width, color="red", label="Target 1")

                                            ax.set_xticks(x)
                                            ax.set_xticklabels(["0", "1"])
                                            ax.set_ylim(0, 100)
                                            ax.set_ylabel("Pourcentage")
                                            ax.set_title(var, fontsize=10)
                                            
                                            # Ajouter la valeur client dans la légende
                                            client_val = features_to_display.get(var)
                                            if isinstance(client_val, (int, float)) and int(client_val) in [0, 1]:
                                                client_val_int = int(client_val)
                                                # Modifier les labels pour inclure la valeur client
                                                legend_labels = ["Target 0", "Target 1", f"Client: {client_val_int}"]
                                                
                                                # Highlighting the client's bar with a border
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
                                                
                                                # Créer une légende personnalisée
                                                from matplotlib.patches import Rectangle
                                                legend_elements = [
                                                    Rectangle((0, 0), 1, 1, facecolor='green', label='Target 0'),
                                                    Rectangle((0, 0), 1, 1, facecolor='red', label='Target 1'),
                                                    Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=3, label=f'Client: {client_val_int}')
                                                ]
                                                ax.legend(handles=legend_elements, fontsize=8)
                                            else:
                                                ax.legend(fontsize=8)

                                            st.pyplot(fig)

                            except FileNotFoundError:
                                st.warning("📁 Fichier 'bool_stats.json' introuvable.")
                            except Exception as e:
                                st.error(f"Erreur lors de l'affichage des booléennes : {e}")


                        elif model is None:
                            st.warning("⚠️ Modèle non disponible pour l'analyse des variables")
                        else:
                            st.warning("⚠️ Fichier d'importance globale non disponible")
                            
                    else:
                        st.warning("La réponse de l'API ne contient pas les clés 'prediction' ou 'probability'.")
                        st.json(prediction_result)
            
            # Affichage des données brutes
            with st.expander("📄 Voir les données brutes du client"):
                st.json(features_to_display)

        else:
            st.warning(f"ID client {client_id} non trouvé dans les données disponibles.")

# Footer
st.markdown("---")
st.markdown("💡 **Astuce**: Si vous rencontrez des problèmes, vérifiez que l'API est bien déployée et accessible.")