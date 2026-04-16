"""Built-up / urbanized density analysis for satellite images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import BUILDING_HIGH_THRESHOLD, BUILDING_LOW_THRESHOLD
from utils.helpers import image_to_rgb_array, mask_to_image, overlay_mask


GREEN_LOWER = np.array([35, 35, 35], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)
WATER_LOWER = np.array([85, 25, 35], dtype=np.uint8)
WATER_UPPER = np.array([132, 255, 255], dtype=np.uint8)

URBAN_GRAY_LOWER = np.array([0, 0, 70], dtype=np.uint8)
URBAN_GRAY_UPPER = np.array([179, 85, 245], dtype=np.uint8)
URBAN_LIGHT_LOWER = np.array([0, 0, 145], dtype=np.uint8)
URBAN_LIGHT_UPPER = np.array([179, 75, 255], dtype=np.uint8)
URBAN_BEIGE_LOWER = np.array([8, 15, 85], dtype=np.uint8)
URBAN_BEIGE_UPPER = np.array([40, 145, 250], dtype=np.uint8)
URBAN_BROWN_GRAY_LOWER = np.array([5, 20, 55], dtype=np.uint8)
URBAN_BROWN_GRAY_UPPER = np.array([32, 140, 225], dtype=np.uint8)

MIN_URBAN_COMPONENT_AREA = 40


@dataclass(frozen=True)
class BuildingAnalysisResult:
    """Result of approximate built-up area analysis on a satellite image."""

    density_index: float
    built_percent: float
    edge_density_percent: float
    contour_count: int
    density_class: str
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


def _filter_urban_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only tiny isolated noise from a broad urbanized-area mask."""

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    kept_count = 0

    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_URBAN_COMPONENT_AREA:
            continue

        cleaned[labels == label] = 255
        kept_count += 1

    return cleaned, kept_count


def analyze_building_density(image: Image.Image) -> BuildingAnalysisResult:
    """Estimate built-up/urbanized density from a satellite image.

    This is an educational heuristic: it does not try to detect exact building
    polygons. It estimates urbanized territory by excluding vegetation and
    water, then finding gray, light, beige and brown-gray artificial surfaces.
    """

    rgb = image_to_rgb_array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    water_mask = cv2.inRange(hsv, WATER_LOWER, WATER_UPPER)

    urban_color_mask = _combine_hsv_ranges(
        hsv,
        (
            (URBAN_GRAY_LOWER, URBAN_GRAY_UPPER),
            (URBAN_LIGHT_LOWER, URBAN_LIGHT_UPPER),
            (URBAN_BEIGE_LOWER, URBAN_BEIGE_UPPER),
            (URBAN_BROWN_GRAY_LOWER, URBAN_BROWN_GRAY_UPPER),
        ),
    )

    excluded = cv2.bitwise_or(green_mask, water_mask)
    urban_mask = cv2.bitwise_and(urban_color_mask, cv2.bitwise_not(excluded))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_OPEN, open_kernel)
    urban_mask, component_count = _filter_urban_components(urban_mask)

    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_gray, threshold1=55, threshold2=150)
    urban_edges = cv2.bitwise_and(edges, urban_mask)

    urban_bool = urban_mask > 0
    built_up_percent = float(urban_bool.mean() * 100.0)
    edge_density_percent = float((urban_edges > 0).mean() * 100.0)
    density_score = float(0.7 * built_up_percent + 0.3 * edge_density_percent)

    if density_score < BUILDING_LOW_THRESHOLD:
        density_class = "низкая"
        conclusion = (
            "Низкая плотность застройки: на спутниковом снимке мало серых, "
            "светлых и бежевых урбанизированных поверхностей после исключения "
            "зелени и воды."
        )
    elif density_score <= BUILDING_HIGH_THRESHOLD:
        density_class = "средняя"
        conclusion = (
            "Средняя плотность застройки: урбанизированные поверхности заметны, "
            "но чередуются с зелёными, водными или открытыми территориями."
        )
    else:
        density_class = "высокая"
        conclusion = (
            "Высокая плотность застройки: значительная часть спутникового "
            "фрагмента похожа на плотную городскую ткань, крыши, асфальт и "
            "другие искусственные поверхности."
        )

    return BuildingAnalysisResult(
        density_index=density_score,
        built_percent=built_up_percent,
        edge_density_percent=edge_density_percent,
        contour_count=component_count,
        density_class=density_class,
        mask_image=mask_to_image(urban_bool),
        overlay_image=overlay_mask(image, urban_bool, color=(220, 70, 60)),
        conclusion=conclusion,
    )
