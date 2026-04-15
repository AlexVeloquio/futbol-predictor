"""Tests para el módulo de predicción — Liga MX."""

import numpy as np
import pytest
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "inference"))
from predict import predict, build_feature_vector, LABELS


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.55, 0.25, 0.20]])
    return model


@pytest.fixture
def mock_metadata():
    return {
        "model_name": "random_forest",
        "features": [
            "h_win_rate", "h_avg_scored", "h_avg_conceded", "h_avg_ht",
            "a_win_rate", "a_avg_scored", "a_avg_conceded", "a_avg_ht",
            "win_rate_diff", "goals_diff", "defense_diff",
        ],
        "teams": ["América", "Guadalajara", "Cruz Azul"],
    }


@pytest.fixture
def mock_team_stats():
    return {
        "América": {"win_rate": 0.7, "avg_scored": 1.8, "avg_conceded": 0.6, "avg_ht": 0.8},
        "Guadalajara": {"win_rate": 0.5, "avg_scored": 1.2, "avg_conceded": 1.0, "avg_ht": 0.5},
        "Cruz Azul": {"win_rate": 0.6, "avg_scored": 1.5, "avg_conceded": 0.8, "avg_ht": 0.7},
    }


class TestBuildFeatureVector:
    def test_correct_length(self, mock_team_stats):
        vector = build_feature_vector("América", "Guadalajara", mock_team_stats)
        assert len(vector) == 11

    def test_win_rate_diff(self, mock_team_stats):
        vector = build_feature_vector("América", "Guadalajara", mock_team_stats)
        expected_diff = 0.7 - 0.5
        assert abs(vector[8] - expected_diff) < 1e-6

    def test_unknown_team_raises(self, mock_team_stats):
        with pytest.raises(ValueError, match="no encontrado"):
            build_feature_vector("Equipo Falso", "América", mock_team_stats)


class TestPredict:
    def test_returns_correct_structure(self, mock_model, mock_metadata, mock_team_stats):
        result = predict("América", "Guadalajara", mock_model, mock_metadata, mock_team_stats)
        assert "home_team" in result
        assert "away_team" in result
        assert "prediction" in result
        assert "label" in result
        assert "probabilities" in result

    def test_includes_team_names(self, mock_model, mock_metadata, mock_team_stats):
        result = predict("América", "Guadalajara", mock_model, mock_metadata, mock_team_stats)
        assert result["home_team"] == "América"
        assert result["away_team"] == "Guadalajara"

    def test_probabilities_sum_to_one(self, mock_model, mock_metadata, mock_team_stats):
        result = predict("América", "Guadalajara", mock_model, mock_metadata, mock_team_stats)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_label_matches_prediction(self, mock_model, mock_metadata, mock_team_stats):
        result = predict("América", "Guadalajara", mock_model, mock_metadata, mock_team_stats)
        assert result["label"] == LABELS[result["prediction"]]
