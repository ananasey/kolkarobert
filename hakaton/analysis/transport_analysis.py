"""Transport infrastructure analysis for Yandex map scheme images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import TRANSPORT_HIGH_THRESHOLD, TRANSPORT_LOW_THRESHOLD
from utils.helpers import image_to_rgb_array, mask_to_image, overlay_mask


MIN_COMPONENT_AREA = 24
MIN_ROAD_AREA = 24
MIN_ROAD_LENGTH = 28
MAX_ICON_AREA = 120
MIN_ROAD_ASPECT_RATIO = 2.0

ROAD_WHITE_GRAY_RANGES = (
    (np.array([0, 0, 205], dtype=np.uint8), np.array([179, 55, 255], dtype=np.uint8)),
    (np.array([0, 0, 178], dtype=np.uint8), np.array([179, 32, 232], dtype=np.uint8)),
)
ROAD_YELLOW_RANGES = (
    (np.array([15, 45, 145], dtype=np.uint8), np.array([38, 255, 255], dtype=np.uint8)),
)
ROAD_ORANGE_RANGES = (
    (np.array([5, 65, 150], dtype=np.uint8), np.array([24, 255, 255], dtype=np.uint8)),
)

GREEN_LOWER = np.array([35, 30, 75], dtype=np.uint8)
GREEN_UPPER = np.array([92, 255, 255], dtype=np.uint8)
WATER_LOWER = np.array([85, 20, 75], dtype=np.uint8)
WATER_UPPER = np.array([132, 255, 255], dtype=np.uint8)
BUILDING_GRAY_LOWER = np.array([0, 0, 130], dtype=np.uint8)
BUILDING_GRAY_UPPER = np.array([179, 38, 212], dtype=np.uint8)
BUILDING_BEIGE_LOWER = np.array([8, 8, 135], dtype=np.uint8)
BUILDING_BEIGE_UPPER = np.array([38, 62, 225], dtype=np.uint8)


@dataclass(frozen=True)
class TransportAnalysisResult:
    """Result of road-only detection on a map scheme."""

    transport_index: float
    transport_percent: float
    long_line_count: int
    mask_image: Image.Image
    overlay_image: Image.Image
    conclusion: str


def _combine_hsv_ranges(
    hsv: np.ndarray,
    ranges: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Combine several HSV ranges into one uint8 mask."""

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def _linear_open(mask: np.ndarray) -> np.ndarray:
    """Keep line-like structures in several directions."""

    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)),
        np.eye(13, dtype=np.uint8),
        np.fliplr(np.eye(13, dtype=np.uint8)),
    )
    linear_mask = np.zeros_like(mask)
    for kernel in kernels:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        linear_mask = cv2.bitwise_or(linear_mask, opened)
    return linear_mask


def _filter_road_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep long road-like components and remove POI icons/text fragments."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    long_component_count = 0
    max_component_area = int(mask.shape[0] * mask.shape[1] * 0.08)

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < max(MIN_COMPONENT_AREA, MIN_ROAD_AREA) or area > max_component_area:
            continue

        aspect_ratio = max(width, height) / max(min(width, height), 1)
        bbox_length = max(width, height)
        fill_ratio = area / max(width * height, 1)
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)

        compact_icon = (
            area <= MAX_ICON_AREA
            and aspect_ratio < 1.65
            and fill_ratio > 0.42
        )
        text_fragment = bbox_length < MIN_ROAD_LENGTH and area < MAX_ICON_AREA
        road_like = (
            bbox_length >= MIN_ROAD_LENGTH
            and (
                aspect_ratio >= MIN_ROAD_ASPECT_RATIO
                or perimeter >= MIN_ROAD_LENGTH * 3.0
                or fill_ratio <= 0.38
            )
        )

        if compact_icon or text_fragment or not road_like:
            continue

        cleaned[labels == label] = 255
        if bbox_length >= MIN_ROAD_LENGTH * 1.5 or perimeter >= MIN_ROAD_LENGTH * 4.0:
            long_component_count += 1

    return cleaned, long_component_count


def analyze_transport_infrastructure(image: Image.Image) -> TransportAnalysisResult:
    """Detect roads on a Yandex map scheme with color, morphology and geometry."""

    rgb = image_to_rgb_array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    white_gray_candidates = _combine_hsv_ranges(hsv, ROAD_WHITE_GRAY_RANGES)
    yellow_roads = _combine_hsv_ranges(hsv, ROAD_YELLOW_RANGES)
    orange_roads = _combine_hsv_ranges(hsv, ROAD_ORANGE_RANGES)

    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    water_mask = cv2.inRange(hsv, WATER_LOWER, WATER_UPPER)
    building_gray = cv2.inRange(hsv, BUILDING_GRAY_LOWER, BUILDING_GRAY_UPPER)
    building_beige = cv2.inRange(hsv, BUILDING_BEIGE_LOWER, BUILDING_BEIGE_UPPER)
    buildings_mask = cv2.bitwise_or(building_gray, building_beige)
    dark_text_mask = cv2.inRange(gray, 0, 95)

    excluded = cv2.bitwise_or(green_mask, water_mask)
    excluded = cv2.bitwise_or(excluded, buildings_mask)
    excluded = cv2.bitwise_or(excluded, dark_text_mask)
    excluded = cv2.dilate(
        excluded,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    edges = cv2.Canny(gray, threshold1=45, threshold2=145)
    edge_support = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    white_gray_roads = cv2.bitwise_and(white_gray_candidates, edge_support)
    white_gray_roads = cv2.bitwise_and(white_gray_roads, cv2.bitwise_not(excluded))
    white_gray_roads = _linear_open(white_gray_roads)

    colored_roads = cv2.bitwise_or(yellow_roads, orange_roads)
    colored_roads = cv2.bitwise_and(colored_roads, cv2.bitwise_not(excluded))
    road_mask = cv2.bitwise_or(colored_roads, white_gray_roads)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, close_kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, open_kernel)
    road_mask = cv2.bitwise_or(road_mask, _linear_open(road_mask))
    road_mask = cv2.dilate(road_mask, open_kernel, iterations=1)

    road_mask, long_line_count = _filter_road_components(road_mask)
    road_bool = road_mask > 0
    road_density = float(road_bool.mean() * 100.0)
    transport_index = road_density

    if road_density < TRANSPORT_LOW_THRESHOLD:
        conclusion = (
            "Низкая транспортная насыщенность: на схеме найдено мало дорожных "
            "линий белого, жёлтого или оранжевого цвета."
        )
    elif road_density <= TRANSPORT_HIGH_THRESHOLD:
        conclusion = (
            "Средняя транспортная насыщенность: дорожная сеть заметна, но не "
            "занимает большую часть фрагмента."
        )
    else:
        conclusion = (
            "Высокая транспортная насыщенность: на схеме выделена плотная сеть "
            "дорог и магистралей."
        )

    return TransportAnalysisResult(
        transport_index=transport_index,
        transport_percent=road_density,
        long_line_count=long_line_count,
        mask_image=mask_to_image(road_bool),
        overlay_image=overlay_mask(image, road_bool, color=(40, 130, 235)),
        conclusion=conclusion,
    )
