"""
handler.py — Entry point de AWS Lambda para el servicio de predicción.

Recibe requests con home_team y away_team, retorna la predicción.
"""

import json
import os
import logging
import boto3
from predict import predict, load_artifacts, get_available_teams

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get("MODEL_BUCKET", "futbol-predictor-models")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "futbol-predictions")

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

MODEL_FILES = {
    "models/model.pkl": "/tmp/model.pkl",
    "models/metadata.json": "/tmp/metadata.json",
    "models/team_stats.json": "/tmp/team_stats.json",
}


def _download_model():
    """Descarga artefactos desde S3 (una vez por cold start)."""
    if not os.path.exists("/tmp/model.pkl"):
        logger.info("Descargando modelo desde S3...")
        for s3_key, local_path in MODEL_FILES.items():
            s3.download_file(S3_BUCKET, s3_key, local_path)
        logger.info("Modelo descargado.")

    return load_artifacts(
        model_path="/tmp/model.pkl",
        meta_path="/tmp/metadata.json",
        stats_path="/tmp/team_stats.json",
    )


def lambda_handler(event, context):
    """Handler principal invocado por API Gateway."""
    try:
        body = json.loads(event.get("body", "{}"))
        home_team = body.get("home_team", "").strip()
        away_team = body.get("away_team", "").strip()

        # Validar input
        if not home_team or not away_team:
            model, metadata, _ = _download_model()
            teams = get_available_teams(metadata)
            return _response(400, {
                "error": "Se requieren 'home_team' y 'away_team'.",
                "available_teams": teams,
            })

        # Cargar modelo y predecir
        model, metadata, team_stats = _download_model()
        result = predict(home_team, away_team, model, metadata, team_stats)

        # Cache en DynamoDB
        _cache_prediction(result)

        return _response(200, result)

    except ValueError as e:
        # Equipo no encontrado
        _, metadata, _ = _download_model()
        return _response(404, {
            "error": str(e),
            "available_teams": get_available_teams(metadata),
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return _response(500, {"error": str(e)})


def _cache_prediction(result: dict):
    """Almacena predicción en DynamoDB."""
    try:
        table = dynamodb.Table(DYNAMO_TABLE)
        table.put_item(Item={
            "match_key": f"{result['home_team']}_vs_{result['away_team']}",
            "prediction": result["label"],
            "probabilities": json.dumps(result["probabilities"]),
            "model": result["model"],
        })
    except Exception as e:
        logger.warning(f"Cache DynamoDB falló: {e}")


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }
