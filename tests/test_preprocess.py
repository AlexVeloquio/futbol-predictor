"""Tests para el módulo de preprocesamiento — Liga MX."""

import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from preprocess import create_target, engineer_features, TARGET_LABELS


@pytest.fixture
def sample_df():
    """DataFrame de ejemplo con formato Liga MX Kaggle."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "date": [
            "2023-01-07", "2023-01-08", "2023-01-14",
            "2023-01-15", "2023-01-21", "2023-01-22",
        ],
        "home_team": ["América", "Guadalajara", "Cruz Azul", "América", "Guadalajara", "Cruz Azul"],
        "away_team": ["Guadalajara", "Cruz Azul", "América", "Cruz Azul", "América", "Guadalajara"],
        "home_goals": [2, 1, 0, 3, 1, 1],
        "away_goals": [1, 1, 2, 0, 1, 2],
        "home_win": [True, False, False, True, False, False],
        "away_win": [False, False, True, False, False, True],
        "home_goals_half_time": [1, 0, 0, 2, 1, 0],
        "away_goals_half_time": [0, 1, 1, 0, 0, 1],
        "home_goals_fulltime": [2, 1, 0, 3, 1, 1],
        "away_goals_fulltime": [1, 1, 2, 0, 1, 2],
        "season": [2023, 2023, 2023, 2023, 2023, 2023],
    })


class TestCreateTarget:
    def test_home_win_maps_to_zero(self, sample_df):
        result = create_target(sample_df)
        assert result.loc[0, "target"] == 0  # América gana en casa

    def test_draw_maps_to_one(self, sample_df):
        result = create_target(sample_df)
        assert result.loc[1, "target"] == 1  # Guadalajara 1-1 Cruz Azul

    def test_away_win_maps_to_two(self, sample_df):
        result = create_target(sample_df)
        assert result.loc[2, "target"] == 2  # América gana de visitante

    def test_target_is_int(self, sample_df):
        result = create_target(sample_df)
        assert result["target"].dtype == int


class TestEngineerFeatures:
    def test_generates_feature_columns(self, sample_df):
        df = create_target(sample_df)
        result = engineer_features(df)
        expected_cols = [
            "h_win_rate", "h_avg_scored", "h_avg_conceded", "h_avg_ht",
            "a_win_rate", "a_avg_scored", "a_avg_conceded", "a_avg_ht",
            "win_rate_diff", "goals_diff", "defense_diff",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Falta columna: {col}"

    def test_filters_no_history(self, sample_df):
        df = create_target(sample_df)
        result = engineer_features(df)
        # Primeros partidos pueden ser filtrados
        assert len(result) <= len(sample_df)

    def test_win_rate_diff_is_correct(self, sample_df):
        df = create_target(sample_df)
        result = engineer_features(df)
        for _, row in result.iterrows():
            assert abs(row["win_rate_diff"] - (row["h_win_rate"] - row["a_win_rate"])) < 1e-6
