"""Matplotlib route visualization."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from services.geocoder import GeoPoint


def render_route_graph(points: list[GeoPoint], route_indices: list[int]) -> Image.Image:
    """Render a labeled route graph as a PNG image."""

    lon_values = np.array([point.lon for point in points], dtype=float)
    lat_values = np.array([point.lat for point in points], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fb")

    ax.scatter(lon_values, lat_values, s=90, color="#1f77b4", zorder=3)

    for order, point_index in enumerate(route_indices[:-1], start=1):
        point = points[point_index]
        ax.annotate(
            f"{order}. {point.name}",
            (point.lon, point.lat),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#1f2933",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d0d7de"},
        )

    for current_index, next_index in zip(route_indices, route_indices[1:]):
        current = points[current_index]
        next_point = points[next_index]
        ax.annotate(
            "",
            xy=(next_point.lon, next_point.lat),
            xytext=(current.lon, current.lat),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2.2,
                "color": "#d62728",
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2,
        )

    ax.set_title("Оптимальный замкнутый маршрут", fontsize=14, pad=14)
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    ax.grid(True, linestyle="--", alpha=0.35)

    lon_padding = max((lon_values.max() - lon_values.min()) * 0.12, 0.5)
    lat_padding = max((lat_values.max() - lat_values.min()) * 0.12, 0.5)
    ax.set_xlim(lon_values.min() - lon_padding, lon_values.max() + lon_padding)
    ax.set_ylim(lat_values.min() - lat_padding, lat_values.max() + lat_padding)

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
