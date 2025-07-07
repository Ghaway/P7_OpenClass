import streamlit as st
import json, requests
import pandas as pd, numpy as np
import shap, matplotlib.pyplot as plt
import plotly.graph_objects as go

# ── 2.4.2 – Configuration de la page ───────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction de Défaut Client ",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🏦 Application de Prédiction de Défaut Client ")

# ── Fichiers & Endpoints ───────────────────────────────────────────────────────
DATA_FILE       = "first_5_rows.json"
FEATURE_FILE    = "feature_importance_global.json"
BOX_FILE        = "boxplot_stats.json"
BOOL_FILE       = "bool_stats.json"
BIVAR_FILE      = "bivariate_discrete.json"
API_PREDICT     = "https://p7-openclass.onrender.com/predict"
API_SHAP_VALUES = "https://p7-openclass.onrender.com/shap_values"

# ── 1.4.4 – Chargement des JSON ────────────────────────────────────────────────
@st.cache_data
def load_json_list(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            st.error(f"{path} ne contient pas une liste JSON.")
            return []
        # Ajout d'un ID si manquant
        for idx, rec in enumerate(data):
            if isinstance(rec, dict) and "id" not in rec:
                rec["id"] = idx + 1
        return data
    except Exception as e:
        st.error(f"Erreur de chargement {path} : {e}")
        return []

clients     = load_json_list(DATA_FILE)
global_imp  = load_json_list(FEATURE_FILE) if False else json.load(open(FEATURE_FILE))  
# (on chargera FEATURE_FILE autrement, car c'est dict)
with open(FEATURE_FILE, "r", encoding="utf-8") as f: global_imp = json.load(f)
with open(BOX_FILE,    "r", encoding="utf-8") as f: boxplot    = json.load(f)
with open(BOOL_FILE,   "r", encoding="utf-8") as f: boolstats  = json.load(f)
with open("bivariate_discrete.json", "r", encoding="utf-8") as f:
    bivar_data = json.load(f)
# ── API call generic ───────────────────────────────────────────────────────────
def call_api(url, payload, timeout=30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API ({url}) : {e}")
        return {}

# ── Fonctions d’affichage ───────────────────────────────────────────────────────

def show_gauge(prob):
    st.subheader("🎯 Score de Risque")
    score, thresh = prob * 100, 49
    low_c, high_c = "#00429d", "#b30000"   # 1.4.3
    bar_c = low_c if score < thresh else high_c
    emoji = "🟢" if score < thresh else "🔴"
    risk_level = "Faible" if score < thresh else "Élevé"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": f"{emoji} Niveau de Risque"},
        delta={"reference": thresh,
               "increasing": {"color": high_c},
               "decreasing": {"color": low_c}},
        gauge={
            "axis": {"range": [0,100], "tickfont": {"size": 14}},
            "bar": {"color": bar_c},
            "steps": [
                {"range": [0, thresh],   "color": "#006400"},
                {"range": [thresh,100],  "color": "#8B0000"}
            ],
            "threshold": {"line": {"color":"#000","width":4}, "value": thresh}
        }
    ))
    fig.update_layout(font={"size":16,"family":"Arial"}, autosize=True)  # 1.4.4

    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le niveau de risque
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilité de Défaut", f"{prob:.1%}")
    with col2:
        st.metric("Seuil", "49%")
    with col3:
        st.metric("Niveau de Risque", f"{emoji} {risk_level}")
    
    
    
    st.caption("Jauge : vert (<49 %) = faible risqué, rouge (≥49 %) = risque élevé.")  # 1.1.1
    st.markdown("🟢 Faible ｜ 🔴 Élevé")  # 1.4.1


def show_importances(client_payload):
    st.subheader("📈 Analyse des Variables")
    res       = call_api(API_SHAP_VALUES, client_payload, timeout=60)
    shap_vals = res.get("shap_values", {})
    base_val  = res.get("expected_value", 0)
    if not shap_vals:
        st.warning("Impossible de récupérer les valeurs SHAP.")
        return

    feats = list(shap_vals.keys())
    arr   = np.array([shap_vals[f] for f in feats])

    # SHAP Waterfall
    st.subheader(" Waterfall Plot SHAP : liste des variables par ordre d'importance")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.tick_params(labelsize=12); ax.set_title("SHAP Waterfall", fontsize=14)
    shap.plots.waterfall(shap.Explanation(values=arr, base_values=base_val, feature_names=feats),
                            max_display=8, show=False)
    st.pyplot(fig); plt.close(fig)
    st.caption("Diagramme en cascade des valeurs SHAP.")  # 1.1.1

    # DataFrame des importances
    abs_imp = {f: abs(v) for f,v in shap_vals.items()}
    total   = sum(abs_imp.values()) or 1
    norm    = {f: abs_imp[f]/total*100 for f in feats}
    df = pd.DataFrame([
        {"Variable":f, "Globale (%)": global_imp.get(f,0), "Locale (%)": norm[f]}
        for f in feats
    ]).sort_values("Globale (%)", ascending=False)

    # Bar chart Globale vs Locale
    st.subheader("📊 Comparaison Importance Globale vs Locale pour chaque variable")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["Variable"], y=df["Globale (%)"], name="Globale (%)",
        marker_color="#003f5c", marker_pattern={"shape":"/"}, opacity=0.9
    ))
    fig2.add_trace(go.Bar(
        x=df["Variable"], y=df["Locale (%)"], name="Locale (%)",
        marker_color="#d62728", marker_pattern={"shape":"x"}, opacity=0.9
    ))
    fig2.update_layout(barmode="group", xaxis_tickangle=-45, font={"size":16})
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Hachurées = globale, croisées = locale.")  # 1.1.1

    st.subheader("📋 Détail des variables")
    st.dataframe(df, use_container_width=True)


def show_population(var, client):
    st.subheader(f"📦 Analyse du client au sein de la population : {var}")
    col1, _ = st.columns([1,2])
    with col1:
        # continue
        if var in boxplot.get("0", {}) and var in boxplot.get("1", {}):
            raw0, raw1 = boxplot["0"][var], boxplot["1"][var]
            stats0 = {"label":"No-def","med":raw0["median"],"q1":raw0["q1"],
                      "q3":raw0["q3"],"whislo":raw0["min"],"whishi":raw0["max"],"fliers":[]}
            stats1 = {"label":"Def",   "med":raw1["median"],"q1":raw1["q1"],
                      "q3":raw1["q3"],"whislo":raw1["min"],"whishi":raw1["max"],"fliers":[]}

            fig, ax = plt.subplots(figsize=(4,3))
            ax.tick_params(labelsize=12); ax.set_title(var, fontsize=14)
            ax.bxp([stats0, stats1], showfliers=False)
            ax.axhline(client[var], color="red", linestyle="--", linewidth=2)
            st.pyplot(fig); plt.close(fig)
            st.caption("Boxplot : défaut=0 vs défaut=1 ; ligne rouge = valeur du client.")  #1.1.1

        # bool
        elif var in boolstats.get("0", {}) and var in boolstats.get("1", {}):
            # Récupère les effectifs ou pourcentages bruts
            t0, t1 = boolstats["0"][var], boolstats["1"][var]
            
            modalities = [0, 1]
            no_def_vals = []
            def_vals    = []
            
            # Calcule, pour chaque modalité, la part No-def vs Def (en %)
            for m in modalities:
                cnt_no = t0.get(str(m), 0)
                cnt_def = t1.get(str(m), 0)
                total = cnt_no + cnt_def or 1
                no_def_vals.append(cnt_no  / total * 100)
                def_vals.append(cnt_def / total * 100)
            
            # Prépare le graphique
            fig, ax = plt.subplots(figsize=(4,3))
            ax.set_title(var, fontsize=14)
            ax.set_ylabel("Répartition (%)")
            ax.tick_params(labelsize=12)
            
            x = np.arange(len(modalities))
            width = 0.6
            
            bars_no = ax.bar(x,               no_def_vals, width,
                            color="#006400", label="No-def")
            bars_de = ax.bar(x,               def_vals,    width,
                            bottom=no_def_vals,
                            color="#8B0000", label="Def")
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(m) for m in modalities])
            ax.legend(
                fontsize=12,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0
            )
            
            # Hachures sur la barre du client
            client_val = client[var]
            idx = modalities.index(client_val)
            for patch in (bars_no[idx], bars_de[idx]):
                patch.set_hatch("///")
                patch.set_edgecolor("black")
            
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Histogramme empilé 0 vs 1 (chaque barre = 100 %) : vert = pas défaut, rouge = défaut.\n "
                "Barre hachurée = groupe du client."
            )


        else:
            st.warning("Variable non disponible pour l’analyse population.")

def show_bivariate(var1, var2, client):
    # Récupère les données du JSON (ordre indifférent)
    st.subheader(f"📦 Analyse bivariée entre {var1} et {var2}")
    col1, _ = st.columns([1,2])
    with col1:
        key = f"{var1}__{var2}"
        data = bivar_data.get(key) or bivar_data.get(f"{var2}__{var1}", [])
        if not data:
            st.warning("Pas de données pour ce couple de variables.")
            return

        # Extrait x, y et statut def
        xs = np.array([d[var1] for d in data])
        ys = np.array([d[var2] for d in data])
        defs = np.array([d["def"] for d in data])

        # Ajout d'un léger jitter pour séparer les points superposés
        jitter = 0.05
        xs = xs + np.random.uniform(-jitter, jitter, len(xs))
        ys = ys + np.random.uniform(-jitter, jitter, len(ys))

        # Sépare indices no-def vs def
        idx0 = np.where(defs == 0)[0]
        idx1 = np.where(defs == 1)[0]

        # Tracé
        fig, ax = plt.subplots(figsize=(5,4))
        ax.set_title(f"{var1} vs {var2}", fontsize=14)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)

        ax.scatter(xs[idx0], ys[idx0], c="#000000", alpha=0.6, label="No-def")
        ax.scatter(xs[idx1], ys[idx1], c="#ff0000", alpha=0.6, label="Def")

        # Point client en évidence
        cx, cy = client[var1], client[var2]
        ax.scatter(
            cx, cy,
            s=150,
            edgecolors="yellow",
            facecolors="none",
            linewidth=2,
            label="Client"
        )

        # Légende en dehors
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=12
        )
        fig.subplots_adjust(right=0.75)

        st.pyplot(fig)
        plt.close(fig)


# ── Sidebar : tests API, sélection client et analyses ─────────────────────────
with st.sidebar:
    st.header("🔍 Connexions & Sélection")
    if st.button("Tester API Prédiction"):
        try:
            r = requests.get(API_PREDICT.replace("/predict","/"), timeout=5)
            st.success("✅ Prédiction OK") if r.status_code==200 else st.error(f"❌ {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    if st.button("Tester API SHAP"):
        try:
            test = {k:0 for k in list(global_imp)[:5]}
            r = requests.post(API_SHAP_VALUES, json=test, timeout=5)
            st.success("✅ SHAP OK") if r.status_code==200 else st.error(f"❌ {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    # Sélection client (repris de votre code d'origine)
    if not clients:
        st.error("Aucun client disponible."); st.stop()
    data_by_id   = {c["id"]:c for c in clients}
    available_ids = sorted(data_by_id.keys())
    client_id    = st.selectbox("Sélection ID client", available_ids)

    st.markdown("---")
    sel_opts = st.multiselect(
        "Analyses disponibles",
        ["Score", "Features Importance", "Population", "Bi-variée"]
    )

# ── Récupération du client hors sidebar ───────────────────────────────────────
client  = data_by_id[client_id]
payload = {k:v for k,v in client.items() if k != "id"}

# ── Exécution des analyses ────────────────────────────────────────────────────
if sel_opts:
    pr   = call_api(API_PREDICT, payload)
    prob = pr.get("probability", 0)
    pred = pr.get("prediction",   0)

    if "Score" in sel_opts:
        show_gauge(prob)
        st.write("✅ Aucun défaut prévu" if pred==0 else "⚠️ Défaut prévu")

    if "Features Importance" in sel_opts:
        show_importances(payload)

    if "Population" in sel_opts:
        # liste des variables continues et booléennes
        cont_vars = [v for v in payload if v in boxplot.get("0", {})]
        bool_vars = [v for v in payload if v in boolstats.get("0", {})]
        pop_vars  = sorted(set(cont_vars + bool_vars))
        var = st.selectbox("Variable population", pop_vars)
        show_population(var, client)
    
    if "Bi-variée" in sel_opts:
        # Liste des features uniques disponibles
        pairs = list(bivar_data.keys())
        features = sorted({v for key in pairs for v in key.split("__")})
        
        var1 = st.selectbox("Variable 1", features, key="biv1")
        var2 = st.selectbox("Variable 2", features, key="biv2")
        
        if var1 and var2 and var1 != var2:
            show_bivariate(var1, var2, client)
    
else:
    st.info("Sélectionnez au moins une analyse dans la sidebar.")

# ── Données brutes ───────────────────────────────────────────────────────────
with st.expander("📄 Données brutes du client"):
    st.json(client)
