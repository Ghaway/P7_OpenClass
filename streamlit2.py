```python
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
    # Couleurs à contraste élevé
    low_color  = "#00429d"  # bleu foncé
    high_color = "#b30000"  # rouge foncé
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
    fig.update_layout(
        font={"size": 16, "family": "Arial"},  # texte redimensionnable
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Figure : jauge interactive montrant le score de risque en pourcentage. "
        "Bleu (<49%) indique faible risque, rouge (≥49%) indique risque élevé."
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

    # Waterfall SHAP
    st.subheader("🔱 Waterfall SHAP")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(values=arr_vals, base_values=base_val, feature_names=features),
        max_display=10, show=False
    )
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Diagramme en cascade des valeurs SHAP, montrant l’impact de chaque variable."
    )

    # Normalisation locale
    abs_imp = {f: abs(v) for f, v in shap_vals.items()}
    total   = sum(abs_imp.values()) or 1
    norm_imp = {f: abs_imp[f] / total * 100 for f in features}

    df = pd.DataFrame([
        {
            "Variable": f,
            "Valeur client": client_payload[f],
            "Importance globale (%)": global_imp.get(f, 0),
            "Importance locale (%)": norm_imp[f]
        }
        for f in features
    ]).sort_values("Importance globale (%)", ascending=False)

    # Barre groupée accessible
    st.subheader("📊 Comparaison Globale vs Locale")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["Variable"], y=df["Importance globale (%)"],
        name="Globale (%)",
        marker_color="#1f77b4",  # bleu
        marker_pattern={"shape": "/"},
        opacity=0.9
    ))
    fig2.add_trace(go.Bar(
        x=df["Variable"], y=df["Importance locale (%)"],
        name="Locale (%)",
        marker_color="#ff7f0e",  # orange
        marker_pattern={"shape": "x"},
        opacity=0.9
    ))
    fig2.update_layout(
        barmode="group",
        xaxis_tickangle=-45,
        font={"size": 16},
        legend={"font": {"size": 14}}
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Histogramme comparant l’importance globale (hachures '/') et locale (motif 'x')."
    )

    # Tableau détaillé
    st.subheader("📋 Tableau détaillé")
    st.dataframe(df, use_container_width=True)

# --- Analyse population accessible ---
def show_population(var, client):
    st.subheader(f"Analyse population pour : {var}")
    if var in boxplot.get("0", {}):
        stats0 = boxplot["0"][var]
        stats1 = boxplot["1"][var]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bxp(
            [dict(label="No-def", **stats0), dict(label="Def", **stats1)],
            showfliers=False
        )
        ax.axhline(client[var], color="red", linestyle="--", linewidth=2)
        ax.set_title(var, fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        st.pyplot(fig)
        st.caption(
            f"Boxplot comparant la distribution de '{var}' pour défaut=0 et défaut=1. "
            "La ligne rouge indique la valeur du client."
        )
    else:
        st.warning("Variable non disponible pour l’analyse population.")

# --- Vérifications initiales ---
if not clients:
    st.stop()

data_by_id = {c["id"]: c for c in clients if isinstance(c, dict) and "id" in c}
if not data_by_id:
    st.error("Aucun client valide chargé (pas d’ID trouvé).")
    st.stop()

# --- Sélection client & options ---
client_id = st.sidebar.selectbox("Sélection Client (ID)", sorted(data_by_id.keys()))
client    = data_by_id[client_id]
sel_opts  = st.sidebar.multiselect(
    "Choisissez vos analyses",
    ["Score", "Importances", "Population"]
)

# --- Exécution des analyses ---
if sel_opts:
    payload = {k: v for k, v in client.items() if k != "id"}
    pr = call_api(API_PREDICT, payload)
    prob = pr.get("probability", 0)
    pred = pr.get("prediction", 0)

    if "Score" in sel_opts:
        show_gauge(prob)
        st.write(
            "✅ Aucun défaut prévu" if pred == 0 else "⚠️ Défaut prévu"
        )

    if "Importances" in sel_opts:
        show_importances(payload)

    if "Population" in sel_opts:
        var = st.selectbox(
            "Variable population",
            [v for v in client if v in boxplot.get("0", {})]
        )
        show_population(var, client)
else:
    st.info("Sélectionnez une ou plusieurs analyses dans la barre latérale.")

# --- Données brutes ---
with st.expander("Données brutes du client"):
    st.write(
        "Les données JSON suivantes correspondent aux features utilisées pour la prédiction."
    )
    st.json(client)
```