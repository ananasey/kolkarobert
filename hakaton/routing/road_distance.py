"""Distance matrix calculation with road API and geodesic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from config import DISTANCE_MATRIX_URL, Settings
from services.geocoder import GeoPoint
from utils.cache import make_cache_key, read_json_cache, write_json_cache
from utils.helpers import haversine_distance_km


class DistanceMatrixError(RuntimeError):
    """Raised when both road and fallback distance calculation fail."""


def _safe_http_error_message(exc: requests.RequestException) -> str:
    """Build an HTTP error message without leaking API keys from request URLs."""

    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}: {response.reason}"
    return exc.__class__.__name__


@dataclass(frozen=True)
class DistanceMatrixResult:
    """Distance matrix plus metadata about road/fallback mode."""

    matrix_km: np.ndarray
    mode_matrix: list[list[str]]
    used_road_api: bool
    used_fallback: bool
    message: str


def _geodesic_matrix(points: list[GeoPoint]) -> np.ndarray:
    size = len(points)
    matrix = np.zeros((size, size), dtype=float)
    for i, origin in enumerate(points):
        for j, destination in enumerate(points):
            if i == j:
                continue
            matrix[i, j] = haversine_distance_km(
                origin.lat,
                origin.lon,
                destination.lat,
                destination.lon,
            )
    return matrix


def _extract_distance_km(element: dict[str, Any]) -> float | None:
    status = str(element.get("status", "")).upper()
    if status and status not in {"OK", "FOUND"}:
        return None

    distance = element.get("distance")
    if isinstance(distance, dict):
        value = distance.get("value")
        if value is not None:
            return float(value) / 1000.0
    if isinstance(distance, (int, float)):
        return float(distance) / 1000.0

    # Some routing APIs use length/value fields. Keep this parser tolerant so
    # minor response-shape differences do not break the educational project.
    for key in ("length", "distanceMeters"):
        value = element.get(key)
        if isinstance(value, dict) and value.get("value") is not None:
            return float(value["value"]) / 1000.0
        if isinstance(value, (int, float)):
            return float(value) / 1000.0
    return None


def _parse_yandex_matrix(payload: dict[str, Any], expected_size: int) -> np.ndarray:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_size:
        raise DistanceMatrixError("Distance Matrix API вернул неожиданную структуру rows.")

    matrix = np.zeros((expected_size, expected_size), dtype=float)
    for row_index, row in enumerate(rows):
        elements = row.get("elements") if isinstance(row, dict) else None
        if not isinstance(elements, list) or len(elements) != expected_size:
            raise DistanceMatrixError(
                "Distance Matrix API вернул неожиданную структуру elements."
            )
        for column_index, element in enumerate(elements):
            if row_index == column_index:
                matrix[row_index, column_index] = 0.0
                continue
            distance_km = _extract_distance_km(element)
            if distance_km is None:
                raise DistanceMatrixError(
                    "Для одной или нескольких пар API не вернул дорожное расстояние."
                )
            matrix[row_index, column_index] = distance_km
    return matrix


def calculate_distance_matrix(
    points: list[GeoPoint],
    settings: Settings,
) -> DistanceMatrixResult:
    """Calculate road distance matrix with honest geodesic fallback."""

    if len(points) < 2:
        raise DistanceMatrixError("Для маршрута нужно минимум два пункта.")

    cache_key = make_cache_key(
        "distance-matrix-road-v2",
        [point.ll for point in points],
        settings.distance_matrix_key,
    )
    cached = read_json_cache(settings.cache_dir, cache_key)
    if cached is not None:
        return DistanceMatrixResult(
            matrix_km=np.array(cached["matrix_km"], dtype=float),
            mode_matrix=cached["mode_matrix"],
            used_road_api=bool(cached["used_road_api"]),
            used_fallback=bool(cached["used_fallback"]),
            message=str(cached["message"]),
        )

    fallback_matrix = _geodesic_matrix(points)
    fallback_modes = [
        ["-" if i == j else "геодезия" for j in range(len(points))]
        for i in range(len(points))
    ]

    if not settings.has_distance_matrix_key:
        message = (
            "Дорожная матрица недоступна: не задан YANDEX_DISTANCE_MATRIX_API_KEY "
            "или общий YANDEX_API_KEY. "
            "Использованы геодезические расстояния."
        )
        return DistanceMatrixResult(fallback_matrix, fallback_modes, False, True, message)

    params = {
        "apikey": settings.distance_matrix_key,
        "origins": "|".join(point.lat_lon for point in points),
        "destinations": "|".join(point.lat_lon for point in points),
        "mode": "driving",
    }

    fallback_reason = ""
    try:
        response = requests.get(
            DISTANCE_MATRIX_URL,
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        road_matrix = _parse_yandex_matrix(payload, len(points))
    except requests.RequestException as exc:
        fallback_reason = _safe_http_error_message(exc)
        message = (
            "Не удалось получить дорожные расстояния через Yandex Distance Matrix API. "
            f"Использован fallback на геодезические расстояния. Причина: {fallback_reason}"
        )
        result = DistanceMatrixResult(fallback_matrix, fallback_modes, False, True, message)
    except (ValueError, DistanceMatrixError) as exc:
        fallback_reason = str(exc)
        message = (
            "Не удалось получить дорожные расстояния через Yandex Distance Matrix API. "
            f"Использован fallback на геодезические расстояния. Причина: {fallback_reason}"
        )
        result = DistanceMatrixResult(fallback_matrix, fallback_modes, False, True, message)
    else:
        road_modes = [
            ["-" if i == j else "дорога" for j in range(len(points))]
            for i in range(len(points))
        ]
        result = DistanceMatrixResult(
            matrix_km=road_matrix,
            mode_matrix=road_modes,
            used_road_api=True,
            used_fallback=False,
            message="Использованы дорожные расстояния Yandex Distance Matrix API.",
        )

    if result.used_road_api:
        write_json_cache(
            settings.cache_dir,
            cache_key,
            {
                "matrix_km": result.matrix_km.tolist(),
                "mode_matrix": result.mode_matrix,
                "used_road_api": result.used_road_api,
                "used_fallback": result.used_fallback,
                "message": result.message,
            },
        )
    return result
