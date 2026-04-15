# Arquitectura del Sistema

## Visión General

El sistema opera en dos fases: desarrollo local (entrenamiento) y despliegue en AWS (solo inferencia).

```
╔═══════════════════════ FASE 1: LOCAL ═══════════════════════╗
║                                                              ║
║  CSV (Kaggle)                                                ║
║      │                                                       ║
║      ▼                                                       ║
║  preprocess.py ──▶ matches.csv (features)                    ║
║                         │                                    ║
║                         ▼                                    ║
║                    train.py                                   ║
║                    ├── model.pkl         (modelo serializado) ║
║                    ├── metadata.json     (config + métricas)  ║
║                    └── team_stats.json   (stats por equipo)   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
                           │
                      make deploy
                           │
╔═══════════════════ FASE 2: AWS (DEPLOY) ════════════════════╗
║                                                              ║
║  ┌──────────┐     ┌─────────────────┐     ┌──────────────┐  ║
║  │    S3    │     │     Lambda      │     │  API Gateway  │  ║
║  │          │────▶│   (Inference)   │◀────│  POST /predict│  ║
║  │model.pkl │     │   ECR Image     │     │               │  ║
║  │stats.json│     └────────┬────────┘     └──────────────┘  ║
║  └──────────┘              │                                 ║
║                            ▼                                 ║
║                      ┌──────────┐                            ║
║                      │ DynamoDB │                            ║
║                      │ (caché)  │                            ║
║                      └──────────┘                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Fase 1 — Pipeline Local

### Preprocesamiento (`preprocess.py`)

1. Lee el CSV crudo de Liga MX desde `data/raw/`.
2. Genera la variable objetivo (`target`) a partir de `home_win` / `away_win`.
3. Calcula features históricas rolling (ventana de 5 partidos) por equipo:
   - Tasa de victorias, promedio de goles anotados/recibidos, goles al medio tiempo.
4. Genera features de diferencia entre local y visitante.
5. Guarda el dataset procesado en `data/processed/matches.csv`.

### Entrenamiento (`train.py`)

1. Carga el dataset procesado.
2. Divide en train/test (80/20 stratificado).
3. Entrena dos modelos:
   - **Random Forest** (principal): 200 árboles, max_depth=15.
   - **Logistic Regression** (baseline): multinomial, 1000 iteraciones.
4. Selecciona el mejor por cross-validation (5 folds).
5. Genera y guarda tres artefactos:
   - `model.pkl` — modelo serializado.
   - `metadata.json` — features, métricas, lista de equipos.
   - `team_stats.json` — estadísticas más recientes de cada equipo.

### Predicción Local

```bash
make predict HOME="América" AWAY="Guadalajara"
```

Usa `predict.py` que:
1. Carga los artefactos de `models/`.
2. Busca las stats de ambos equipos en `team_stats.json`.
3. Construye el feature vector y ejecuta la predicción.

## Fase 2 — Deploy AWS

### Amazon S3
Almacena los tres artefactos del modelo. La Lambda los descarga al `/tmp/` en el cold start.

### AWS Lambda + ECR
Una sola función Lambda empaquetada como imagen Docker (para superar el límite de 250 MB con scikit-learn). Recibe requests, carga modelo, predice y responde.

### API Gateway
Endpoint REST público: `POST /predict` con body `{"home_team": "...", "away_team": "..."}`.

### DynamoDB
Caché de predicciones ya procesadas con latencia < 10ms.

## CI/CD — GitHub Actions

| Pipeline | Trigger         | Pasos                              |
|----------|-----------------|------------------------------------|
| CI       | push / PR       | Lint → Test → Docker Build         |
| CD       | merge a main    | Build → Push ECR → Update Lambda   |

## Decisiones Técnicas

| Decisión                    | Justificación                                               |
|-----------------------------|-------------------------------------------------------------|
| Entrenamiento local         | Sin créditos AWS para training; el dataset es pequeño       |
| Solo inference en Lambda    | Minimiza costos; pago por invocación                        |
| Docker en Lambda (ECR)      | scikit-learn + pandas exceden 250 MB                        |
| team_stats.json             | Permite predecir con solo nombres de equipos                |
| Rolling window = 5          | Balance entre recencia y estabilidad estadística            |
| Random Forest + Log. Reg.   | RF es robusto para tabular; LR como baseline interpretable  |
