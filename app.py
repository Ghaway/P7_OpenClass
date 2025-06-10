from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import json
import os
import traceback
import logging
import shap # Importez SHAP ici
from sklearn.pipeline import Pipeline # Importez Pipeline pour gérer les modèles scikit-learn
import numpy as np

# === Configuration ===
MODEL_PATH = os.environ.get('MODEL_PATH', 'file://mlflow_model')
FEATURES_PATH = os.environ.get('FEATURES_PATH', 'feature_names.json')
THRESHOLD = float(os.environ.get('THRESHOLD', 0.49))

# === Initialisation du logging ===
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
    logger.error(traceback.format_exc())

# === Chargement des noms de features ===
FEATURE_NAMES = []
try:
    logger.info(f"Attempting to load feature names from {FEATURES_PATH}")
    with open(FEATURES_PATH, 'r') as f:
        FEATURE_NAMES = json.load(f)
    logger.info(f"Successfully loaded feature names from {FEATURES_PATH}")
except FileNotFoundError:
    logger.error(f"Error: {FEATURES_PATH} not found.")
except json.JSONDecodeError:
    logger.error(f"Error: Could not decode JSON from {FEATURES_PATH}. Check file syntax.")
    logger.error(traceback.format_exc())
except Exception as e:
    logger.error(f"Error loading feature names: {e}")
    logger.error(traceback.format_exc())

@app.route("/")
def index():
    return jsonify({"message": "✅ API de prédiction de défaut client en ligne."})

@app.route("/predict", methods=["POST"])
def predict():
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

        if not all(feature in data for feature in FEATURE_NAMES):
            missing_features = [feature for feature in FEATURE_NAMES if feature not in data]
            logger.error(f"Missing features in incoming data: {missing_features}")
            return jsonify({
                "error": "Les données d'entrée sont incomplètes.",
                "attendu": FEATURE_NAMES,
                "recu": list(data.keys()),
                "manquantes": missing_features
            }), 400

        try:
            input_data = pd.DataFrame([data]).reindex(columns=FEATURE_NAMES, fill_value=0)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Erreur lors de la création du DataFrame à partir des données fournies."}), 400

        logger.info(f"DataFrame created for prediction:\n{input_data}")

        try:
            probabilities = model.predict_proba(input_data)
            default_probability = probabilities[:, 1][0]
            prediction = int(default_probability > THRESHOLD)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Une erreur est survenue lors de la prédiction: {e}"}), 500

        return jsonify({
            "prediction": prediction,
            "probability": float(default_probability)
        })

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Une erreur est survenue: {e}"}), 500

@app.route("/shap_values", methods=["POST"]) # Nouvelle route pour SHAP
def get_shap_values():
    if model is None or not FEATURE_NAMES:
        error_message = ""
        if model is None:
            error_message += "Model failed to load. "
        if not FEATURE_NAMES:
            error_message += "Feature names failed to load."
        logger.error(f"API SHAP not fully initialized: {error_message}")
        return jsonify({"error": f"API SHAP not fully initialized: {error_message}"}), 500

    try:
        data = request.get_json()
        if data is None:
            logger.error("Received None data from request.get_json() for SHAP.")
            return jsonify({"error": "La requête SHAP ne contient pas de JSON valide."}), 400

        if not all(feature in data for feature in FEATURE_NAMES):
            missing_features = [feature for feature in FEATURE_NAMES if feature not in data]
            logger.error(f"Missing features in incoming data for SHAP: {missing_features}")
            return jsonify({
                "error": "Les données d'entrée SHAP sont incomplètes.",
                "attendu": FEATURE_NAMES,
                "recu": list(data.keys()),
                "manquantes": missing_features
            }), 400
        
        # Ensure data order matches FEATURE_NAMES
        # Use pandas DataFrame to ensure correct feature order and handling missing values
        input_data = pd.DataFrame([data]).reindex(columns=FEATURE_NAMES, fill_value=0)
        
        logger.info(f"DataFrame created for SHAP calculation:\n{input_data}")

        # === Calcul des valeurs SHAP ===
        try:
            # Check if the model is a Pipeline and extract the base model
            if isinstance(model, Pipeline):
                base_model = model.steps[-1][1]
                # Apply all transformations except the last step (the model)
                transformed_input_data = input_data
                for step_name, transformer in model.steps[:-1]:
                    if hasattr(transformer, 'transform'):
                        transformed_input_data = transformer.transform(transformed_input_data)
            else:
                base_model = model
                transformed_input_data = input_data # No transformation needed

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(transformed_input_data)

            # Gérer le cas où shap_values peut être une liste (classification binaire)
            if isinstance(shap_values, list):
                # Pour la classification binaire, prendre les valeurs de la classe positive (index 1)
                if len(shap_values) == 2:
                    shap_values_to_return = shap_values[1][0].tolist() # Première observation, classe positive
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                else: # Fallback for other list scenarios (e.g., multi-class where you want first class)
                    shap_values_to_return = shap_values[0][0].tolist()
                    expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:  # (n_samples, n_features, n_classes)
                shap_values_to_return = shap_values[0, :, 1].tolist()  # Première observation, classe positive
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:  # (n_samples, n_features)
                shap_values_to_return = shap_values[0, :].tolist()  # Première observation
                expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:  # Array 1D or unexpected format
                shap_values_to_return = shap_values.tolist()
                expected_value = explainer.expected_value # This might need adjustment based on how explainer.expected_value behaves for 1D

            shap_dict = dict(zip(FEATURE_NAMES, shap_values_to_return))

            # Retourner à la fois les valeurs SHAP et la expected_value
            return jsonify({
                "shap_values": shap_dict,
                "expected_value": float(expected_value) if expected_value is not None else None
            })

        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Erreur lors du calcul des valeurs SHAP: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"An error occurred in SHAP endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Une erreur est survenue dans l'endpoint SHAP: {e}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))