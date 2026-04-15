# Datos — Predictor de Resultados Liga MX

## Fuente

Dataset de Kaggle: [LigaMX Matches 2016-2024](https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022)  
Autor: Gerardo Jaime Escareo

## Descarga

1. Descargar el CSV desde Kaggle (requiere cuenta gratuita).
2. Colocar el archivo en `data/raw/`.

```bash
# O usando la CLI de Kaggle:
pip install kaggle
kaggle datasets download -d gerardojaimeescareo/ligamx-matches-2016-2022 -p data/raw/ --unzip
```

## Diccionario de datos

| Columna                  | Descripción                       | Tipo   |
|--------------------------|-----------------------------------|--------|
| id                       | ID del partido                    | int    |
| referee                  | Nombre del árbitro                | string |
| timezone                 | Zona horaria de la fecha          | string |
| date                     | Fecha del partido                 | string |
| venue_id                 | ID del estadio                    | int    |
| venue_name               | Nombre del estadio                | string |
| venue_city               | Ciudad del estadio                | string |
| season                   | Año de la temporada               | int    |
| round                    | Tipo de ronda                     | string |
| home_team                | Equipo local                      | string |
| away_team                | Equipo visitante                  | string |
| home_win                 | True si ganó el local             | bool   |
| away_win                 | True si ganó el visitante         | bool   |
| home_goals               | Goles del local                   | int    |
| away_goals               | Goles del visitante               | int    |
| home_goals_half_time     | Goles local al medio tiempo       | int    |
| away_goals_half_time     | Goles visitante al medio tiempo   | int    |
| home_goals_fulltime      | Goles local tiempo completo       | int    |
| away_goals_fulltime      | Goles visitante tiempo completo   | int    |
| home_goals_extra_time    | Goles local tiempo extra          | int    |
| away_goals_extratime     | Goles visitante tiempo extra      | int    |
| home_goals_penalty       | Goles local en penales            | int    |
| away_goals_penalty       | Goles visitante en penales        | int    |

## Variable objetivo

Derivada de `home_win` y `away_win`:

| Valor | Etiqueta  | Condición                            |
|-------|-----------|--------------------------------------|
| 0     | Home Win  | home_win == True                     |
| 1     | Draw      | home_win == False y away_win == False|
| 2     | Away Win  | away_win == True                     |

## Features generadas (preprocessing)

| Feature          | Descripción                                        |
|------------------|----------------------------------------------------|
| h_win_rate       | Tasa de victorias recientes del equipo local        |
| h_avg_scored     | Promedio de goles anotados (local, últimos 5)       |
| h_avg_conceded   | Promedio de goles recibidos (local, últimos 5)      |
| h_avg_ht         | Promedio de goles al medio tiempo (local)           |
| a_win_rate       | Tasa de victorias recientes del visitante           |
| a_avg_scored     | Promedio de goles anotados (visitante, últimos 5)   |
| a_avg_conceded   | Promedio de goles recibidos (visitante, últimos 5)  |
| a_avg_ht         | Promedio de goles al medio tiempo (visitante)       |
| win_rate_diff    | h_win_rate - a_win_rate                            |
| goals_diff       | h_avg_scored - a_avg_scored                        |
| defense_diff     | h_avg_conceded - a_avg_conceded                    |

## Estructura

```
data/
├── raw/            # CSV descargado de Kaggle (no se commitea)
├── processed/      # Dataset con features generadas
└── README.md
```
