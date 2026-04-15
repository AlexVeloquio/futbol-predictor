"""
predict.py — Lógica de inferencia: recibe dos equipos y predice el resultado.

Usa el modelo entrenado + team_stats.json para construir el feature vector
a partir de los nombres de los equipos.
"""

import json
import os
import joblib
import numpy as np

LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

_model = None
_metadata = None
_team_stats = None


def load_artifacts(
    model_path: str = "models/model.pkl",
    meta_path: str = "models/metadata.json",
    stats_path: str = "models/team_stats.json",
):
    """Carga modelo, metadatos y stats de equipos (singleton)."""
    global _model, _metadata, _team_stats
    if _model is None:
        _model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            _metadata = json.load(f)
        with open(stats_path, "r") as f:
            _team_stats = json.load(f)
    return _model, _metadata, _team_stats


def get_available_teams(metadata: dict = None) -> list:
    """Retorna la lista de equipos disponibles."""
    if metadata is None:
        _, metadata, _ = load_artifacts()
    return metadata.get("teams", [])


def build_feature_vector(home_team: str, away_team: str, team_stats: dict) -> list:
    """
    Construye el vector de features para un partido dado
    usando las estadísticas históricas de ambos equipos.
    """
    h = team_stats.get(home_team)
    a = team_stats.get(away_team)

    if h is None:
        raise ValueError(f"Equipo local '{home_team}' no encontrado. Verifica el nombre exacto.")
    if a is None:
        raise ValueError(f"Equipo visitante '{away_team}' no encontrado. Verifica el nombre exacto.")

    return [
        h["win_rate"],
        h["avg_scored"],
        h["avg_conceded"],
        h["avg_ht"],
        a["win_rate"],
        a["avg_scored"],
        a["avg_conceded"],
        a["avg_ht"],
        h["win_rate"] - a["win_rate"],         # win_rate_diff
        h["avg_scored"] - a["avg_scored"],     # goals_diff
        h["avg_conceded"] - a["avg_conceded"], # defense_diff
    ]


def predict(home_team: str, away_team: str, model=None, metadata=None, team_stats=None) -> dict:
    """
    Predice el resultado de un partido entre dos equipos de la Liga MX.

    Args:
        home_team: nombre del equipo local (ej: "América")
        away_team: nombre del equipo visitante (ej: "Guadalajara")

    Returns:
        dict con predicción, probabilidades y detalles.
    """
    if model is None or metadata is None or team_stats is None:
        model, metadata, team_stats = load_artifacts()

    feature_vector = build_feature_vector(home_team, away_team, team_stats)
    X = np.array([feature_vector])

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0].tolist()

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "label": LABELS.get(prediction, "Unknown"),
        "probabilities": {
            LABELS[i]: round(p, 4) for i, p in enumerate(probabilities)
        },
        "model": metadata.get("model_name", "unknown"),
    }
