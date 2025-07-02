import streamlit as st
import json, requests
import pandas as pd, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Configurations ---
API_PREDICT = "https://p7-openclass.onrender.com/predict"
API_SHAP    = "https://p7-openclass.onrender.com/shap_values"
DATA_FILES  = {
    "clients": "first_5_rows.json",
    "global_imp": "feature_importance_global.json",
    "boxplot": "boxplot_stats.json",
    "bool":    "bool_stats.json"
}

# --- Caching des chargements JSON ---
@st.cache_data
def load_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}

clients = load_json(DATA_FILES["clients"])
global_imp = load_json(DATA_FILES["global_imp"])
boxplot   = load_json(DATA_FILES["boxplot"])
boolstats = load_json(DATA_FILES["bool"])

# --- Appel API générique ---
def call_api(url, payload, timeout=30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({url}): {e}")
        return {}

# --- Affichage jauge de risque ---
def show_gauge(p):
    score = p * 100
    thresh = 49
    color = "green" if score < thresh else "red"
    emoji = "🟢" if score < thresh else "🔴"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": f"{emoji} Risque"},
        delta={"reference": thresh},
        gauge={
            "axis": {"range": [0,100]},
            "bar": {"color": color},
            "steps":[{"range":[0,thresh],"color":"lightgreen"},
                     {"range":[thresh,100],"color":"lightcoral"}]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# --- SHAP waterfall + tableau des importances ---
def show_importances(client):
    res = call_api(API_SHAP, client, timeout=60)
    vals = res.get("shap_values", {})
    base = res.get("expected_value", 0)
    if not vals: return
    fnames = list(vals)
    shap_vals = np.array([vals[f] for f in fnames])
    st.subheader("🔱 SHAP Waterfall")
    fig, ax = plt.subplots(figsize=(8,4))
    shap.plots.waterfall(shap.Explanation(
        values=shap_vals, base_values=base, feature_names=fnames
    ), max_display=10, show=False)
    st.pyplot(fig); plt.close()

    abs_imp = {f:abs(v) for f,v in vals.items()}
    norm = {f:imp/sum(abs_imp.values())*100 for f,imp in abs_imp.items()}
    df = pd.DataFrame([
        {"Variable":f,
         "Client": client[f],
         "Globale(%)": global_imp.get(f,0),
         "Locale(%)": norm.get(f,0)}
    ] for f in client).sort_values("Globale(%)", ascending=False)
    st.subheader("📊 Globale vs Locale")
    fig = go.Figure()
    for col, color in zip(["Globale(%)","Locale(%)"],["lightblue","orange"]):
        fig.add_trace(go.Bar(x=df.Variable, y=df[col], name=col, marker_color=color))
    fig.update_layout(barmode="group", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

# --- Sélection du client ---
st.title("🏦 Prédiction de Défaut Client")
if not clients:
    st.error("Fichier clients introuvable ou vide")
    st.stop()

data_by_id = {c["id"]:c for c in clients}
cid = st.sidebar.selectbox("Client ID", sorted(data_by_id))
client = data_by_id[cid]
st.header(f"Analyse Client #{cid}")

# --- Options ---
opts = st.sidebar.multiselect("Analyses", 
    ["Score", "Importances", "Population"], default=[]
)

# --- Prédiction & affichage ---
if opts:
    pred_res = call_api(API_PREDICT, {k:v for k,v in client.items() if k!="id"})
    prob = pred_res.get("probability", 0)
    pred = pred_res.get("prediction", 0)
    if "Score" in opts:
        show_gauge(prob)
        st.write(("✅ Pas de défaut","⚠️ Défaut")[pred])
    if "Importances" in opts:
        show_importances(client)
    if "Population" in opts:
        var = st.selectbox("Variable population", 
                           [v for v in client if v in boxplot.get("0",{})])
        if var in boxplot["0"]:
            st.subheader(var)
            df_box = [
                dict(label="No-def", **boxplot["0"][var]),
                dict(label="Def",    **boxplot["1"][var])
            ]
            fig, ax = plt.subplots()
            ax.bxp(df_box, showfliers=False)
            ax.axhline(client[var], color="red", linestyle="--")
            st.pyplot(fig); plt.close()
else:
    st.info("Cochez les analyses à réaliser dans la barre latérale")

# --- Données brutes ---
with st.expander("Voir données brutes"):
    st.json(client)
