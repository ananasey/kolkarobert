"""Satellite green area analysis with OpenCV and HSV segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import GREEN_LOW_THRESHOLD, GREEN_MEDIUM_THRESHOLD
from utils.helpers import image_area_km2, image_to_rgb_array, mask_to_image, overlay_mask


GREEN_HSV_LOWER = np.array([35, 35, 35], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([90, 255, 255], dtype=np.uint8)
MORPH_KERNEL_SIZE = (5, 5)


@dataclass(frozen=True)
class GreenAnalysisResult:
    """Result of satellite vegetation detection."""

    green_share: float
    green_percent: float
    green_area_km2: float
    total_area_km2: float
    mask_image: Image.Image
    overlay_image: Image.Image
    conclusion: str


def analyze_green_areas(
    image: Image.Image,
    *,
    latitude: float,
    zoom: int,
) -> GreenAnalysisResult:
    """Detect vegetation on a satellite image using HSV color segmentation."""

    rgb = image_to_rgb_array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    vegetation_balance = (green >= red * 0.85) & (green >= blue * 0.90)
    green_mask = np.where(vegetation_balance, green_mask, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    mask_bool = green_mask > 0
    green_percent = float(mask_bool.mean() * 100.0)
    total_area_km2 = image_area_km2(image, latitude, zoom)
    green_area_km2 = total_area_km2 * (green_percent / 100.0)

    if green_percent < GREEN_LOW_THRESHOLD:
        conclusion = (
            "Низкое озеленение: на спутниковом снимке зелёные участки занимают "
            "небольшую долю территории."
        )
    elif green_percent < GREEN_MEDIUM_THRESHOLD:
        conclusion = (
            "Среднее озеленение: зелёные зоны заметны, но не доминируют в "
            "пределах выбранного фрагмента."
        )
    else:
        conclusion = (
            "Высокое озеленение: значительная часть спутникового снимка занята "
            "растительностью, парками, лесными массивами или зелёными дворами."
        )

    return GreenAnalysisResult(
        green_share=green_percent / 100.0,
        green_percent=green_percent,
        green_area_km2=green_area_km2,
        total_area_km2=total_area_km2,
        mask_image=mask_to_image(mask_bool),
        overlay_image=overlay_mask(image, mask_bool, color=(0, 190, 70)),
        conclusion=conclusion,
    )
