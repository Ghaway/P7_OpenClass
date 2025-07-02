import streamlit as st
import json, requests
import pandas as pd, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Configuration de la page (WCAG 2.4.2) ---
st.set_page_config(
    page_title="Prédiction de Risque - Accessible",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("🏦 Prédiction de Défaut Client (Accessible)")

# --- API endpoints ---
API_PREDICT = "https://p7-openclass.onrender.com/predict"
API_SHAP    = "https://p7-openclass.onrender.com/shap_values"

# --- Chargement des données clients ---
@st.cache_data
def load_clients(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, list):
            st.error(f"Le fichier {path} ne contient pas une liste de clients.")
            return []
        return data
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Erreur JSON dans : {path}")
        return []
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement de {path} : {e}")
        return []

# --- Chargement des JSON d’importances et stats ---
@st.cache_data
def load_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

clients    = load_clients("first_5_rows.json")
global_imp = load_json("feature_importance_global.json")
boxplot    = load_json("boxplot_stats.json")
boolstats  = load_json("bool_stats.json")

# --- Appel API générique ---
def call_api(url, payload, timeout=30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API ({url}) : {e}")
        return {}

# --- Jauge de risque accessible (1.1.1, 1.4.1, 1.4.3, 1.4.4) ---
def show_gauge(prob):
    score = prob * 100
    thresh = 49
    low_color  = "#00429d"  # contraste élevé
    high_color = "#b30000"
    bar_color  = low_color if score < thresh else high_color
    emoji      = "🟢" if score < thresh else "🔴"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": f"{emoji} Niveau de Risque"},
        delta={
            "reference": thresh,
            "increasing": {"color": high_color},
            "decreasing": {"color": low_color}
        },
        gauge={
            "axis": {"range": [0,100], "tickfont": {"size": 14}},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, thresh], "color": "#c6dbef"},
                {"range": [thresh, 100], "color": "#fcbba1"}
            ],
            "threshold": {"line": {"color": "#000000", "width": 4}, "value": thresh}
        }
    ))
    fig.update_layout(font={"size": 16, "family": "Arial"}, autosize=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Figure : jauge interactive affichant le score de risque (%) ; bleu (<49%) = faible, rouge (≥49%) = élevé."
    )

# --- Importances SHAP + comparatif accessible ---
def show_importances(client_payload):
    res = call_api(API_SHAP, client_payload, timeout=60)
    shap_vals = res.get("shap_values", {})
    base_val  = res.get("expected_value", 0)
    if not shap_vals:
        st.warning("Impossible de calculer les valeurs SHAP.")
        return

    features = list(shap_vals.keys())
    arr_vals = np.array([shap_vals[f] for f in features])

    st.subheader("🔱 Waterfall SHAP")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(values=arr_vals, base_values=base_val, feature_names=features),
        max_display=10, show=False
    )
    st.pyplot