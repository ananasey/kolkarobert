"""Small helper functions shared by the application modules."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
from PIL import Image


EARTH_RADIUS_KM = 6371.0088
WEB_MERCATOR_INITIAL_RESOLUTION = 156543.03392


def normalize_place_name(value: str) -> str:
    """Normalize a place name for duplicate checks and cache keys."""

    cleaned = re.sub(r"\s+", " ", value.strip())
    return cleaned.casefold()


def parse_places(raw_text: str) -> list[str]:
    """Parse user input with one settlement per line."""

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def has_duplicates(values: Iterable[str]) -> bool:
    """Return True when normalized strings contain duplicates."""

    normalized = [normalize_place_name(value) for value in values]
    return len(normalized) != len(set(normalized))


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate geodesic distance between two WGS84 points in kilometers."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert any Pillow image to a uint8 RGB array."""

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def meters_per_pixel(latitude: float, zoom: int) -> float:
    """Estimate ground resolution for a Web Mercator tile at a latitude."""

    latitude_rad = math.radians(latitude)
    return WEB_MERCATOR_INITIAL_RESOLUTION * math.cos(latitude_rad) / (2**zoom)


def image_area_km2(image: Image.Image, latitude: float, zoom: int) -> float:
    """Estimate covered area of a static map image in square kilometers."""

    resolution = meters_per_pixel(latitude, zoom)
    return image.width * image.height * (resolution**2) / 1_000_000.0


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean mask to a black-white image."""

    mask_uint8 = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a boolean mask over an image using the provided RGB color."""

    base = image_to_rgb_array(image).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    mask_3d = np.repeat(mask[:, :, None], 3, axis=2)
    blended = np.where(mask_3d, base * (1.0 - alpha) + overlay * alpha, base)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
