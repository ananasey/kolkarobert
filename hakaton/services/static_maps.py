"""Yandex Static Maps API client."""

from __future__ import annotations

from io import BytesIO
from typing import Literal

import requests
from PIL import Image

from config import DEFAULT_MAP_SIZE, DEFAULT_ZOOM, STATIC_MAPS_URL, Settings
from services.geocoder import GeoPoint
from utils.cache import make_cache_key, read_bytes_cache, write_bytes_cache


class StaticMapError(RuntimeError):
    """Raised when a static map image cannot be downloaded or decoded."""


def _safe_http_error_message(exc: requests.RequestException) -> str:
    """Build an HTTP error message without leaking API keys from request URLs."""

    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}: {response.reason}"
    return exc.__class__.__name__


def load_static_map(
    center: GeoPoint,
    settings: Settings,
    *,
    zoom: int = DEFAULT_ZOOM,
    size: tuple[int, int] = DEFAULT_MAP_SIZE,
    layer: Literal["sat", "map"] = "map",
) -> Image.Image:
    """Download a static map image fragment from Yandex Static Maps API."""

    cache_key = make_cache_key("static-map", center.ll, zoom, size, layer)
    suffix = ".jpg" if layer == "sat" else ".png"
    cached = read_bytes_cache(settings.cache_dir, cache_key, suffix=suffix)
    if cached is not None:
        return Image.open(BytesIO(cached)).convert("RGB")

    params = {
        "ll": center.ll,
        "z": zoom,
        "size": f"{size[0]},{size[1]}",
        "l": layer,
        "lang": "ru_RU",
    }

    try:
        response = requests.get(
            STATIC_MAPS_URL,
            params=params,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StaticMapError(
            "Ошибка HTTP при загрузке карты: "
            f"{_safe_http_error_message(exc)}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        message = response.text[:300].replace("\n", " ")
        raise StaticMapError(f"Static Maps API не вернул изображение: {message}")

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except OSError as exc:
        raise StaticMapError("Не удалось прочитать изображение карты.") from exc

    write_bytes_cache(settings.cache_dir, cache_key, response.content, suffix=suffix)
    return image
