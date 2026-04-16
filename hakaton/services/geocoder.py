"""Yandex Geocoder API client."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import requests

from config import GEOCODER_URL, Settings
from utils.cache import make_cache_key, read_json_cache, write_json_cache


class GeocodingError(RuntimeError):
    """Raised when a place cannot be geocoded reliably."""


def _safe_http_error_message(exc: requests.RequestException) -> str:
    """Build an HTTP error message without leaking API keys from request URLs."""

    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}: {response.reason}"
    return exc.__class__.__name__


@dataclass(frozen=True)
class GeoPoint:
    """A geocoded point in WGS84 coordinates."""

    name: str
    address: str
    lon: float
    lat: float
    precision: str

    @property
    def ll(self) -> str:
        """Coordinates in lon,lat format used by Yandex APIs."""

        return f"{self.lon:.6f},{self.lat:.6f}"

    @property
    def lat_lon(self) -> str:
        """Coordinates in lat,lon format used by Yandex routing APIs."""

        return f"{self.lat:.6f},{self.lon:.6f}"


def _extract_feature(feature: dict[str, Any]) -> GeoPoint:
    geo_object = feature.get("GeoObject", {})
    meta = geo_object.get("metaDataProperty", {}).get("GeocoderMetaData", {})
    point = geo_object.get("Point", {}).get("pos")
    if not point:
        raise GeocodingError("В ответе геокодера нет координат.")

    lon_text, lat_text = point.split()
    return GeoPoint(
        name=geo_object.get("name") or meta.get("text") or "Неизвестный пункт",
        address=meta.get("text") or geo_object.get("description") or "",
        lon=float(lon_text),
        lat=float(lat_text),
        precision=meta.get("precision") or "unknown",
    )


def _decode_cached_points(data: dict[str, Any]) -> list[GeoPoint]:
    return [GeoPoint(**item) for item in data.get("points", [])]


def geocode_place(
    query: str,
    settings: Settings,
    *,
    results: int = 5,
    restrict_to_russia: bool = True,
) -> list[GeoPoint]:
    """Geocode a settlement or city using Yandex Geocoder API.

    The function returns several candidates so the UI can warn users about
    ambiguous geocoding results while still using the first, most relevant one.
    """

    if not settings.has_geocoder_key:
        raise GeocodingError(
            "Не задан YANDEX_GEOCODER_API_KEY или общий YANDEX_API_KEY в файле .env."
        )

    normalized_query = query.strip()
    if not normalized_query:
        raise GeocodingError("Пустое название населённого пункта.")

    search_query = (
        f"{normalized_query}, Россия" if restrict_to_russia else normalized_query
    )
    cache_key = make_cache_key("geocoder", search_query, results)
    cached = read_json_cache(settings.cache_dir, cache_key)
    if cached is not None:
        return _decode_cached_points(cached)

    params = {
        "apikey": settings.geocoder_key,
        "geocode": search_query,
        "format": "json",
        "lang": "ru_RU",
        "results": results,
    }

    try:
        response = requests.get(
            GEOCODER_URL,
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GeocodingError(
            "Ошибка HTTP при обращении к геокодеру: "
            f"{_safe_http_error_message(exc)}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingError("Геокодер вернул не JSON-ответ.") from exc

    collection = (
        payload.get("response", {})
        .get("GeoObjectCollection", {})
        .get("featureMember", [])
    )
    if not collection:
        raise GeocodingError(f"Город или населённый пункт не найден: {query}")

    points = [_extract_feature(feature) for feature in collection]
    write_json_cache(
        settings.cache_dir,
        cache_key,
        {"points": [asdict(point) for point in points]},
    )
    return points
