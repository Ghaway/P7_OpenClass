
# 🧠 Prédiction de Défaut Client – Projet Data Science

## 🎯 Objectif

Ce projet vise à prédire si un client fera défaut sur un prêt bancaire à partir de données personnelles et financières. Il inclut :

- Un modèle de machine learning entraîné avec suivi via **MLflow**
- Une **API Flask** exposant le modèle
- Une **interface utilisateur Streamlit** pour la visualisation et l’interprétation des prédictions

---

## 🗂️ Structure du projet

```
projet/
├── app.py                  # API Flask exposant le modèle MLflow
├── app_streamlit.py        # Interface utilisateur Streamlit
├── feature_names.json      # Liste des variables utilisées par le modèle
├── first_5_rows.json       # Échantillon de données pour tests via Streamlit
├── requirements.txt        # Dépendances Python
├── README.md               # Ce fichier
└── mlflow_model/           # Modèle sauvegardé par MLflow
```

---

## 🔁 Suivi d’expériences avec MLflow

Le modèle est suivi avec **MLflow** :

- Entraînement et sauvegarde du modèle avec `mlflow.sklearn.log_model()`
- Stockage des paramètres, métriques et artefacts dans un répertoire `mlruns/`
- Chargement du modèle dans `app.py` via `mlflow.sklearn.load_model`

---

## ⚙️ API Flask

Le fichier `app.py` propose deux routes :

- `GET /` : Vérification de l’état de l’API
- `POST /predict` : Prend en entrée un JSON avec les features, retourne :
  - `prediction` (0 ou 1)
  - `probability` (probabilité de défaut)

### Exemple d’appel :

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_client.json
```

---

## 🌐 Interface utilisateur Streamlit

Le fichier `app_streamlit.py` permet de :

- Sélectionner un client à partir d’un fichier JSON
- Afficher les variables
- Envoyer les données à l’API
- Afficher la prédiction (défaut ou non) et la probabilité associée

---

## 🚀 Installation et exécution

### 1. Installation des dépendances

```bash
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` sous Windows
pip install -r requirements.txt
```

### 2. Lancer l’API Flask

```bash
export MODEL_PATH="file://mlflow_model"      # ou définir via .env
export FEATURES_PATH="feature_names.json"
python app.py
```

### 3. Lancer l’application Streamlit

```bash
streamlit run app_streamlit.py
```

---

## 📦 Bibliothèques utilisées

- `scikit-learn` – Entraînement du modèle
- `mlflow` – Tracking des expériences et gestion du modèle
- `flask` – Création de l’API
- `streamlit` – Interface utilisateur
- `pandas`, `requests`, `json`, `logging`

---

## 🛠️ Variables d’environnement

- `MODEL_PATH` : chemin vers le modèle MLflow (ex : `file://mlflow_model`)
- `FEATURES_PATH` : chemin vers le fichier JSON listant les variables
- `THRESHOLD` : seuil de décision pour la classification (par défaut : 0.49)
- `PORT` : port d’exposition de l’API Flask (par défaut : 5000)

---

## ✅ Statut

Projet fonctionnel en local et prêt pour un déploiement sur Render. 
