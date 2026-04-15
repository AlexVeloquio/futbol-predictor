"""
train.py — Entrenamiento del modelo de predicción Liga MX.

Entrena Random Forest (principal) y Logistic Regression (baseline),
selecciona el mejor por cross-validation y serializa modelo + stats de equipos.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

FEATURE_COLS = [
    "h_win_rate", "h_avg_scored", "h_avg_conceded", "h_avg_ht",
    "a_win_rate", "a_avg_scored", "a_avg_conceded", "a_avg_ht",
    "win_rate_diff", "goals_diff", "defense_diff",
]
TARGET_COL = "target"
TARGET_NAMES = ["Home Win", "Draw", "Away Win"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset procesado."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}. ¿Corriste preprocess.py?")
    return df


def compute_latest_team_stats(df: pd.DataFrame) -> dict:
    """
    Calcula las estadísticas más recientes de cada equipo
    para poder hacer predicciones en inferencia.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)

    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    latest = {}

    for team in teams:
        # Partidos como local
        home = df[df["home_team"] == team].tail(5)
        # Partidos como visitante
        away = df[df["away_team"] == team].tail(5)

        if len(home) > 0:
            h_stats = home.iloc[-1]
            latest[team] = {
                "win_rate": float(h_stats.get("h_win_rate", 0)),
                "avg_scored": float(h_stats.get("h_avg_scored", 0)),
                "avg_conceded": float(h_stats.get("h_avg_conceded", 0)),
                "avg_ht": float(h_stats.get("h_avg_ht", 0)),
            }
        elif len(away) > 0:
            a_stats = away.iloc[-1]
            latest[team] = {
                "win_rate": float(a_stats.get("a_win_rate", 0)),
                "avg_scored": float(a_stats.get("a_avg_scored", 0)),
                "avg_conceded": float(a_stats.get("a_avg_conceded", 0)),
                "avg_ht": float(a_stats.get("a_avg_ht", 0)),
            }
        else:
            latest[team] = {
                "win_rate": 0.0, "avg_scored": 0.0,
                "avg_conceded": 0.0, "avg_ht": 0.0,
            }

    return latest


def train_models(X_train, X_test, y_train, y_test):
    """Entrena ambos modelos y retorna el mejor."""
    results = {}

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial")
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5, scoring="accuracy").mean()
    results["logistic_regression"] = {"model": lr, "test_acc": lr_acc, "cv_acc": lr_cv}
    print(f"📊 Logistic Regression — Test: {lr_acc:.4f} | CV: {lr_cv:.4f}")

    # Principal: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy").mean()
    results["random_forest"] = {"model": rf, "test_acc": rf_acc, "cv_acc": rf_cv}
    print(f"📊 Random Forest     — Test: {rf_acc:.4f} | CV: {rf_cv:.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_acc"])
    print(f"\n🏆 Mejor modelo: {best_name}")
    return results[best_name]["model"], best_name, results


def save_artifacts(model, model_name, features, metrics, team_stats):
    """Serializa modelo, metadatos y stats de equipos."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    print(f"💾 Modelo guardado en models/model.pkl")

    metadata = {
        "model_name": model_name,
        "features": features,
        "target_names": TARGET_NAMES,
        "teams": sorted(team_stats.keys()),
        "metrics": {
            name: {"test_accuracy": m["test_acc"], "cv_accuracy": m["cv_acc"]}
            for name, m in metrics.items()
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(MODELS_DIR, "team_stats.json"), "w") as f:
        json.dump(team_stats, f, indent=2, ensure_ascii=False)

    print(f"📋 Metadatos y stats de {len(team_stats)} equipos guardados.")


def main():
    data_path = os.path.join(PROCESSED_DIR, "matches.csv")
    print(f"📂 Cargando {data_path}...")
    df = load_data(data_path)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    print(f"   Samples: {len(X)} | Features: {len(FEATURE_COLS)} | Clases: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    print("\n🧠 Entrenando modelos...\n")
    best_model, best_name, all_results = train_models(X_train, X_test, y_train, y_test)

    print("\n📝 Classification Report:\n")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    print("📊 Calculando stats de equipos...")
    team_stats = compute_latest_team_stats(df)

    save_artifacts(best_model, best_name, FEATURE_COLS, all_results, team_stats)
    print("\n✅ Entrenamiento completado.")


if __name__ == "__main__":
    main()
