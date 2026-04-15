# ⚽ Predictor de Resultados — Liga MX

> Modelo de ML para predecir resultados de partidos de la Liga MX (Victoria Local, Empate, Victoria Visitante), desplegado como API serverless en AWS.

**Materia:** Infraestructura y Desarrollo Continuo — ITESO, 8vo Semestre  
**Equipo:** Alejandro Rodríguez Veloquio · Diego Rosales Rojas

---

## Descripción

Sistema que utiliza datos históricos de la Liga MX (2016-2024) para entrenar un modelo de clasificación que predice el resultado de un partido dados dos equipos. El desarrollo y entrenamiento se realizan localmente, y al final se despliega el endpoint de inferencia en AWS Lambda (serverless).

**Flujo:**

1. Se preprocesan los datos y se generan features históricas por equipo.
2. Se entrenan Random Forest y Logistic Regression, se selecciona el mejor.
3. El modelo entrenado se empaqueta en Docker y se despliega en Lambda + API Gateway.
4. El usuario envía `home_team` y `away_team` al endpoint y recibe la predicción.

## Arquitectura

```
┌───────────────────────── Local ──────────────────────────┐
│                                                           │
│  data/raw/         preprocess.py        train.py          │
│  (CSV Kaggle) ───▶ (features) ───▶ (modelo .pkl)         │
│                                    (team_stats.json)      │
└──────────────────────────┬────────────────────────────────┘
                           │  make deploy
                           ▼
┌───────────────────────── AWS ─────────────────────────────┐
│                                                           │
│  S3 (modelo)     Lambda (inference)     API Gateway       │
│  model.pkl ────▶ predict() ◀──────── POST /predict       │
│  team_stats     handler.py              ▼                 │
│                      │             JSON response          │
│                      ▼                                    │
│                  DynamoDB (caché)                          │
└───────────────────────────────────────────────────────────┘
```

Ver [docs/architecture.md](docs/architecture.md) para detalle completo.

## Dataset

**Fuente:** [Liga MX Matches 2016-2024 (Kaggle)](https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022)

Contiene partidos históricos con goles, resultados, equipos, estadios y árbitros. Ver [data/README.md](data/README.md) para el diccionario de datos.

## Tech Stack

| Capa           | Tecnología                           |
|----------------|--------------------------------------|
| ML             | scikit-learn, pandas, numpy          |
| Contenedores   | Docker, Amazon ECR                   |
| Cómputo        | AWS Lambda (container image)         |
| Almacenamiento | Amazon S3                            |
| Base de datos  | Amazon DynamoDB                      |
| API            | Amazon API Gateway (REST)            |
| IaC            | Terraform                            |
| CI/CD          | GitHub Actions                       |

## Estructura del Proyecto

```
.
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml               # Lint → Test → Docker Build
│   └── cd.yml               # Build → Push ECR → Update Lambda
├── data/
│   ├── raw/                 # CSV de Kaggle (no se commitea)
│   └── processed/           # Dataset con features generadas
├── src/
│   ├── training/            # Preprocesamiento + entrenamiento
│   │   ├── preprocess.py
│   │   └── train.py
│   └── inference/           # Servicio de predicción (Lambda)
│       ├── Dockerfile
│       ├── handler.py
│       └── predict.py
├── tests/                   # Tests unitarios (pytest)
├── models/                  # Artefactos entrenados (no se commitean)
├── notebooks/               # EDA
├── infrastructure/          # Terraform (IaC)
├── Makefile                 # Automatización
└── docker-compose.yml       # Entorno local con LocalStack
```

## Inicio Rápido

```bash
# 1. Clonar y preparar entorno
git clone https://github.com/<tu-usuario>/futbol-predictor.git
cd futbol-predictor
make setup
source venv/bin/activate

# 2. Descargar dataset de Kaggle y colocarlo en data/raw/
#    https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022

# 3. Pipeline completo
make pipeline    # preprocess + train

# 4. Probar predicción local
make predict HOME="América" AWAY="Guadalajara"

# 5. Tests
make test
```

## Uso de la API (post-deploy)

```bash
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "América", "away_team": "Guadalajara"}'
```

Respuesta:
```json
{
  "home_team": "América",
  "away_team": "Guadalajara",
  "prediction": 0,
  "label": "Home Win",
  "probabilities": {
    "Home Win": 0.55,
    "Draw": 0.25,
    "Away Win": 0.20
  },
  "model": "random_forest"
}
```

## Modelos

| Modelo                   | Rol       | Justificación                                   |
|--------------------------|-----------|--------------------------------------------------|
| Random Forest Classifier | Principal | Robusto, maneja múltiples features, bajo overfit |
| Logistic Regression      | Baseline  | Rápido, interpretable, punto de comparación      |

## Licencia

MIT
