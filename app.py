from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import os
import traceback
import logging

# === Configuration ===
# Utiliser des variables d'environnement et des valeurs par défaut pour la configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'file://mlflow_model')
FEATURES_PATH = os.environ.get('FEATURES_PATH', 'feature_names.json')
THRESHOLD = float(os.environ.get('THRESHOLD', 0.49))  # Assurez-vous que THRESHOLD est un float

# === Initialisation du logging ===
# Configurez le logging pour une meilleure visibilité des erreurs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Initialisation Flask ===
app = Flask(__name__)

# === Chargement du modèle MLflow ===
model = None
try:
    logger.info(f"Attempting to load MLflow model from {MODEL_PATH}")
    model = mlflow.sklearn.load_model(MODEL_PATH)
    logger.info(f"Successfully loaded MLflow model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading MLflow model from {MODEL_PATH}: {e}")
    logger.error(traceback.format_exc())  # Log the traceback
    # Il est crucial de décider comment gérer cet échec.  Ici, nous continuons,
    # mais l'API ne fonctionnera pas correctement.  Une meilleure approche
    # pourrait être de quitter l'application si le modèle ne se charge pas.
    # raise  # Décommentez ceci si vous voulez que l'application s'arrête en cas d'erreur de chargement du modèle

# === Chargement des noms de features ===
FEATURE_NAMES = []
try:
    logger.info(f"Attempting to load feature names from {FEATURES_PATH}")
    with open(FEATURES_PATH, 'r') as f:
        FEATURE_NAMES = json.load(f)
    logger.info(f"Successfully loaded feature names from {FEATURES_PATH}")
except FileNotFoundError:
    logger.error(f"Error: {FEATURES_PATH} not found.")
    # Ne pas continuer si le fichier des features est introuvable.
    # Sans les noms de features, l'API ne peut pas fonctionner correctement.
    # Vous pourriez vouloir quitter l'application ici.
    # sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Error: Could not decode JSON from {FEATURES_PATH}. Check file syntax.")
    logger.error(traceback.format_exc())
    # Ne pas continuer en cas d'erreur de JSON.
    # sys.exit(1)
except Exception as e:
    logger.error(f"Error loading feature names: {e}")
    logger.error(traceback.format_exc())
    # Gérer toute autre exception inattendue.
    # sys.exit(1)

@app.route("/")
def index():
    return jsonify({"message": "✅ API de prédiction de défaut client en ligne."})

@app.route("/predict", methods=["POST"])
def predict():
    # Vérifier si le modèle et les noms de features ont été chargés avec succès
    if model is None or not FEATURE_NAMES:
        error_message = ""
        if model is None:
            error_message += "Model failed to load. "
        if not FEATURE_NAMES:
            error_message += "Feature names failed to load."
        logger.error(f"API not fully initialized: {error_message}")
        return jsonify({"error": f"API not fully initialized: {error_message}"}), 500

    try:
        data = request.get_json()
        if data is None:
            logger.error("Received None data from request.get_json()")
            return jsonify({"error": "La requête ne contient pas de JSON valide."}), 400

        # Vérifie que toutes les features attendues sont présentes
        if not all(feature in data for feature in FEATURE_NAMES):
            missing_features = [
                feature for feature in FEATURE_NAMES if feature not in data]
            logger.error(f"Missing features in incoming data: {missing_features}")
            logger.error(f"Expected features: {FEATURE_NAMES}")
            logger.error(f"Received data keys: {list(data.keys())}")
            return jsonify({
                "error": "Les données d'entrée sont incomplètes.",
                "attendu": FEATURE_NAMES,
                "recu": list(data.keys()),
                "manquantes": missing_features
            }), 400

        # Convertir les données en DataFrame pandas
        try:
            input_data = pd.DataFrame([data]).reindex(columns=FEATURE_NAMES,
                                                    fill_value=0)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Erreur lors de la création du DataFrame à partir des données fournies."}), 400

        logger.info(f"DataFrame created for prediction:\n{input_data}")

        # Effectuer la prédiction
        try:
            probabilities = model.predict_proba(input_data)
            logger.info(f"Prediction probabilities: {probabilities}")
            default_probability = probabilities[:, 1][0]
            logger.info(f"Default probability: {default_probability}")
            prediction = int(default_probability > THRESHOLD)
            logger.info(
                f"Prediction (based on threshold {THRESHOLD}): {prediction}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Une erreur est survenue lors de la prédiction: {e}"}), 500

        return jsonify({
            "prediction": prediction,
            "probability": float(default_probability)
        })

    except Exception as e:
        # Gérer les autres erreurs potentielles pendant la prédiction
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Une erreur est survenue: {e}"}), 500
if __name__ == "__main__":
    # Pour une utilisation en production, utilisez un serveur WSGI comme Gunicorn ou uWSGI
    # app.run(debug=True) # Utile pour le développement, mais pas en production.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
