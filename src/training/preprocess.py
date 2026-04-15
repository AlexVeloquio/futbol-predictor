"""
preprocess.py — Limpieza y feature engineering del dataset Liga MX (Kaggle).

Lee el CSV crudo desde data/raw/, genera features históricas por equipo
y guarda el dataset listo para entrenamiento en data/processed/.
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

TARGET_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def load_raw(raw_dir: str) -> pd.DataFrame:
    """Carga el CSV de Liga MX."""
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en {raw_dir}")

    frames = []
    for f in csv_files:
        path = os.path.join(raw_dir, f)
        df = pd.read_csv(path)
        print(f"  📄 {f}: {len(df)} registros")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Genera la variable objetivo a partir de home_win / away_win."""
    conditions = [
        df["home_win"] == True,
        df["away_win"] == True,
    ]
    choices = [0, 2]
    df["target"] = np.select(conditions, choices, default=1)
    df["target"] = df["target"].astype(int)
    return df


def compute_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un historial unificado por equipo (como local y visitante).
    """
    records = []

    for _, row in df.iterrows():
        date = row["date"]
        records.append({
            "date": date,
            "team": row["home_team"],
            "goals_scored": row["home_goals"],
            "goals_conceded": row["away_goals"],
            "goals_ht": row.get("home_goals_half_time", 0),
            "won": 1 if row.get("home_win") else 0,
            "is_home": 1,
        })
        records.append({
            "date": date,
            "team": row["away_team"],
            "goals_scored": row["away_goals"],
            "goals_conceded": row["home_goals"],
            "goals_ht": row.get("away_goals_half_time", 0),
            "won": 1 if row.get("away_win") else 0,
            "is_home": 0,
        })

    history = pd.DataFrame(records)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history.sort_values(["team", "date"], inplace=True)
    return history


def build_rolling_stats(history: pd.DataFrame, window: int = 5) -> dict:
    """Calcula rolling stats por equipo sobre los últimos N partidos."""
    stats = {}
    for team, group in history.groupby("team"):
        g = group.copy()
        g["win_rate"] = g["won"].rolling(window, min_periods=1).mean()
        g["avg_scored"] = g["goals_scored"].rolling(window, min_periods=1).mean()
        g["avg_conceded"] = g["goals_conceded"].rolling(window, min_periods=1).mean()
        g["avg_ht"] = g["goals_ht"].rolling(window, min_periods=1).mean()
        g["home_pct"] = g["is_home"].rolling(window, min_periods=1).mean()
        stats[team] = g.reset_index(drop=True)
    return stats


def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Genera features para cada partido basadas en historial previo."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    history = compute_team_history(df)
    team_stats = build_rolling_stats(history, window=window)

    team_counters = {team: 0 for team in team_stats}
    rows = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hi, ai = team_counters.get(home, 0), team_counters.get(away, 0)

        def get_stats(team, idx):
            if team in team_stats and idx > 0:
                s = team_stats[team].iloc[idx - 1]
                return s["win_rate"], s["avg_scored"], s["avg_conceded"], s["avg_ht"]
            return 0.0, 0.0, 0.0, 0.0

        h_wr, h_gs, h_gc, h_ht = get_stats(home, hi)
        a_wr, a_gs, a_gc, a_ht = get_stats(away, ai)

        rows.append({
            "h_win_rate": h_wr,
            "h_avg_scored": h_gs,
            "h_avg_conceded": h_gc,
            "h_avg_ht": h_ht,
            "a_win_rate": a_wr,
            "a_avg_scored": a_gs,
            "a_avg_conceded": a_gc,
            "a_avg_ht": a_ht,
            "win_rate_diff": h_wr - a_wr,
            "goals_diff": h_gs - a_gs,
            "defense_diff": h_gc - a_gc,
        })

        team_counters[home] = hi + 1
        team_counters[away] = ai + 1

    feat_df = pd.DataFrame(rows)
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # Filtrar partidos sin historial
    before = len(result)
    result = result[result["h_win_rate"] + result["a_win_rate"] > 0].copy()
    print(f"   Filtrados {before - len(result)} partidos sin historial previo.")

    return result


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📂 Cargando dataset Liga MX...")
    df = load_raw(RAW_DIR)
    print(f"   Total: {len(df)} partidos\n")

    print("🏷️  Generando variable objetivo...")
    df = create_target(df)
    for t, label in TARGET_LABELS.items():
        count = (df["target"] == t).sum()
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\n🔧 Feature engineering (window=5)...")
    df = engineer_features(df, window=5)

    out_path = os.path.join(PROCESSED_DIR, "matches.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Guardado: {out_path} ({len(df)} registros)")


if __name__ == "__main__":
    main()
