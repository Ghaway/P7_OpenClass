import pytest
import json
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, Mock

from app import app as flask_app

# ======== 1. TESTS DE L'API FLASK ========

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get('/')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert "message" in data
    assert "API de prédiction de défaut client en ligne" in data["message"]


@pytest.fixture
def valid_client_data():
    return {
        "NAME_INCOME_TYPE_Working": 1,
        "EXT_SOURCE_2": 0.65,
        "NAME_EDUCATION_TYPE_Higher education": 1,
        "NAME_EDUCATION_TYPE_Secondary / secondary special": 0,
        "cc_PERIODE_Y_sum_sum": 3,
        "FLAG_EMP_PHONE": 1,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_3": 0.7,
        "FLAG_DOCUMENT_3": 1,
        "CODE_GENDER": 1,
        "FLAG_OWN_CAR": 0
    }

@pytest.fixture
def invalid_client_data():
    return {
        "NAME_INCOME_TYPE_Working": 1,
        "EXT_SOURCE_2": 0.65
    }


@patch('app.model')
@patch('app.FEATURE_NAMES', [
    "NAME_INCOME_TYPE_Working", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Secondary / secondary special", "cc_PERIODE_Y_sum_sum",
    "FLAG_EMP_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_3", "FLAG_DOCUMENT_3",
    "CODE_GENDER", "FLAG_OWN_CAR"
])
def test_predict_valid_data(mock_model, client, valid_client_data):
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    response = client.post('/predict', json=valid_client_data, content_type='application/json')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] == 1
    assert data["probability"] == 0.7


@patch('app.model')
@patch('app.FEATURE_NAMES', [
    "NAME_INCOME_TYPE_Working", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Secondary / secondary special", "cc_PERIODE_Y_sum_sum",
    "FLAG_EMP_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_3", "FLAG_DOCUMENT_3",
    "CODE_GENDER", "FLAG_OWN_CAR"
])
def test_predict_invalid_data(mock_model, client, invalid_client_data):
    response = client.post('/predict', json=invalid_client_data, content_type='application/json')
    data = json.loads(response.data)
    assert response.status_code == 400
    assert "error" in data
    assert "incomplètes" in data["error"]  # Corrigé


def test_predict_empty_request(client):
    response = client.post('/predict', json={}, content_type='application/json')
    assert response.status_code == 400


# ======== 2. TESTS DES FONCTIONS STREAMLIT ========

@pytest.fixture
def mock_streamlit_error():
    with patch('streamlit.error') as mock:
        yield mock


def test_load_data_file_not_found(mock_streamlit_error):
    with patch('builtins.open', side_effect=FileNotFoundError()):
        from app_streamlit import load_data
        result = load_data('non_existent_file.json')
        assert result is None
        assert mock_streamlit_error.call_count >= 1
        assert any("n'a pas été trouvé" in str(call.args[0]) for call in mock_streamlit_error.call_args_list)


def test_load_data_invalid_json(mock_streamlit_error):
    with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
        with patch('json.load', side_effect=json.JSONDecodeError("Test error", "", 0)):
            from app_streamlit import load_data
            result = load_data('test.json')
            assert result is None
            assert mock_streamlit_error.call_count >= 1
            assert any("Impossible de décoder" in str(call.args[0]) for call in mock_streamlit_error.call_args_list)


import importlib

def test_load_data_valid_json():
    test_data = [{
        "NAME_INCOME_TYPE_Working": 1,
        "EXT_SOURCE_2": 0.65,
        "NAME_EDUCATION_TYPE_Higher education": 1,
        "NAME_EDUCATION_TYPE_Secondary / secondary special": 0,
        "cc_PERIODE_Y_sum_sum": 3,
        "FLAG_EMP_PHONE": 1,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_3": 0.7,
        "FLAG_DOCUMENT_3": 1,
        "CODE_GENDER": 1,
        "FLAG_OWN_CAR": 0
    }]
    
    with patch('builtins.open', mock_open(read_data=json.dumps(test_data))):
        with patch('json.load', return_value=test_data):
            import app_streamlit
            importlib.reload(app_streamlit)  # <- Force rechargement après les mocks
            result = app_streamlit.load_data("test.json")
            
            assert result is not None
            assert len(result) == 1
            assert result[0]["id"] == 1
            assert result[0]["NAME_INCOME_TYPE_Working"] == 1
