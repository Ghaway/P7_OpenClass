import json
import numpy as np
from app import app

def load_features():
    with open("mlflow_model/feature_names.json") as f:
        return json.load(f)

def generate_mock_payload(feature_names):
    return {feature: 1.0 for feature in feature_names}  # Tous à 1.0 pour le test

def test_predict_endpoint():
    client = app.test_client()
    
    feature_names = load_features()
    payload = generate_mock_payload(feature_names)

    response = client.post("/predict", data=json.dumps(payload),
                           content_type="application/json")
    print("🔎 Contenu de la réponse :", response.get_data(as_text=True))


    assert response.status_code == 200, f"Statut HTTP inattendu : {response.status_code}"
    result = response.get_json()

    assert "probabilite_defaut" in result, "La réponse JSON ne contient pas 'probabilite_defaut'"
    assert "decision" in result, "La réponse JSON ne contient pas 'decision'"
    assert isinstance(result["probabilite_defaut"], float), "Le champ 'probabilite_defaut' doit être un float"
    assert result["decision"] in ["Accepté", "Refusé"], "La décision doit être 'Accepté' ou 'Refusé'"
