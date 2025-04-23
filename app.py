from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import os

# === Configuration ===
MODEL_PATH = "mlflow_model"
FEATURES_PATH = os.path.join(MODEL_PATH, "feature_names.json")
THRESHOLD = 0.42  # Seuil métier optimisé

# === Initialisation Flask ===
app = Flask(__name__)

# === Chargement du modèle MLflow ===
model = mlflow.sklearn.load_model(MODEL_PATH)

# === Chargement des noms de features ===
with open(FEATURES_PATH) as f:
    FEATURE_NAMES = json.load(f)

@app.route("/")
def index():
    return jsonify({"message": "✅ API de prédiction de défaut client en ligne."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Vérifie que toutes les features attendues sont présentes
        if not all(feature in data for feature in FEATURE_NAMES):
            return jsonify({
                "error": "Les données d'entrée sont incomplètes.",
                "attendu": FEATURE_NAMES,
                "recu": list(data.keys())
            }), 400

        # ✅ Création d'un DataFrame pour éviter les warnings sklearn
        X = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Prédiction
        prob = float(model.predict_proba(X)[0][1])  # cast numpy -> float
        decision = "Accepté" if prob < THRESHOLD else "Refusé"

        return jsonify({
            "probabilite_defaut": round(prob, 4),
            "decision": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
