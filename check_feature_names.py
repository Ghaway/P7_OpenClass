import mlflow.sklearn
import json
import os

MODEL_PATH = "mlflow_model"
OUTPUT_PATH = os.path.join(MODEL_PATH, "feature_names.json")

def extract_feature_names(model):
    """
    Essaie d'extraire les noms de colonnes utilisés à l'entraînement.
    Supporte les pipelines scikit-learn avec transformers.
    """
    try:
        return list(model.feature_names_in_)  # cas simple
    except AttributeError:
        pass

    # Si c'est un pipeline
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                return list(step.get_feature_names_out())

    raise ValueError("Impossible d'extraire les noms de features depuis ce modèle.")

def main():
    print(f"📦 Chargement du modèle depuis {MODEL_PATH}...")
    model = mlflow.sklearn.load_model(MODEL_PATH)

    try:
        features = extract_feature_names(model)
        print(f"✅ {len(features)} features détectées :")
        for f in features[:20]:
            print(" -", f)
        if len(features) > 20:
            print("   ... (troncature pour affichage)")

        # Enregistrer ?
        save = input("💾 Voulez-vous sauvegarder ces features dans feature_names.json ? (y/n) ").strip().lower()
        if save == "y":
            with open(OUTPUT_PATH, "w") as f:
                json.dump(features, f)
            print(f"✅ Fichier sauvegardé dans {OUTPUT_PATH}")
        else:
            print("❌ Annulé.")

    except Exception as e:
        print("❌ Erreur :", e)

if __name__ == "__main__":
    main()
