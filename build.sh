set -e  # Cette ligne fait échouer le script si une commande échoue

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests..."
pytest

echo "All tests passed! Ready for deployment."
# Vous pouvez ajouter d'autres commandes de préparation au déploiement ici