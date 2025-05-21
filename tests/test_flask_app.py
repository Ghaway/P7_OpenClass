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
    assert "incomplètes" in data["error"]


@patch('app.model', MagicMock())
@patch('app.FEATURE_NAMES', [
    "NAME_INCOME_TYPE_Working", "EXT_SOURCE_2", "NAME_EDUCATION_TYPE_Higher education",
    "NAME_EDUCATION_TYPE_Secondary / secondary special", "cc_PERIODE_Y_sum_sum",
    "FLAG_EMP_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_3", "FLAG_DOCUMENT_3",
    "CODE_GENDER", "FLAG_OWN_CAR"
])
def test_predict_empty_request(client):
    response = client.post('/predict', json={}, content_type='application/json')
    data = json.loads(response.data)
    assert response.status_code == 400
    assert "error" in data