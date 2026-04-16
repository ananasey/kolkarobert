"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

GEOCODER_URL = "https://geocode-maps.yandex.ru/v1/"
STATIC_MAPS_URL = "https://static-maps.yandex.ru/1.x/"
DISTANCE_MATRIX_URL = "https://api.routing.yandex.net/v2/distancematrix"

DEFAULT_MAP_SIZE = (650, 450)
DEFAULT_ZOOM = 13
REQUEST_TIMEOUT_SECONDS = 20

GREEN_LOW_THRESHOLD = 8.0
GREEN_MEDIUM_THRESHOLD = 18.0

BUILDING_LOW_THRESHOLD = 18.0
BUILDING_HIGH_THRESHOLD = 35.0

TRANSPORT_LOW_THRESHOLD = 1.5
TRANSPORT_HIGH_THRESHOLD = 4.0


@dataclass(frozen=True)
class Settings:
    """Runtime settings for API access and local storage."""

    yandex_api_key: str
    yandex_geocoder_api_key: str
    yandex_static_maps_api_key: str
    yandex_distance_matrix_api_key: str
    cache_dir: Path
    request_timeout: int

    @property
    def has_yandex_key(self) -> bool:
        """Return True when a non-empty Yandex API key is configured."""

        return bool(self.yandex_api_key.strip())

    @property
    def geocoder_key(self) -> str:
        """Return key for Yandex Geocoder API with fallback to common key."""

        return self.yandex_geocoder_api_key or self.yandex_api_key

    @property
    def static_maps_key(self) -> str:
        """Return key for Yandex Static Maps API with fallback to common key."""

        return self.yandex_static_maps_api_key or self.yandex_api_key

    @property
    def distance_matrix_key(self) -> str:
        """Return key for Yandex Distance Matrix API with fallback to common key."""

        return self.yandex_distance_matrix_api_key or self.yandex_api_key

    @property
    def has_geocoder_key(self) -> bool:
        """Return True when a geocoder key is configured."""

        return bool(self.geocoder_key.strip())

    @property
    def has_static_maps_key(self) -> bool:
        """Return True when a static maps key is configured."""

        return bool(self.static_maps_key.strip())

    @property
    def has_distance_matrix_key(self) -> bool:
        """Return True when a distance matrix key is configured."""

        return bool(self.distance_matrix_key.strip())


def get_settings() -> Settings:
    """Load settings from .env and environment variables."""

    load_dotenv(BASE_DIR / ".env")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Settings(
        yandex_api_key=os.getenv("YANDEX_API_KEY", "").strip(),
        yandex_geocoder_api_key=os.getenv("YANDEX_GEOCODER_API_KEY", "").strip(),
        yandex_static_maps_api_key=os.getenv("YANDEX_STATIC_MAPS_API_KEY", "").strip(),
        yandex_distance_matrix_api_key=os.getenv(
            "YANDEX_DISTANCE_MATRIX_API_KEY",
            "",
        ).strip(),
        cache_dir=CACHE_DIR,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
    )
