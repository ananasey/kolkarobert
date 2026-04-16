"""Exact TSP solver for seven settlements."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class TspResult:
    """Optimal cyclic route returned by the exact solver."""

    route_indices: list[int]
    total_distance_km: float
    total_cost: float


def solve_tsp_exact(
    distance_matrix_km: np.ndarray,
    start_index: int,
    tariff: float,
) -> TspResult:
    """Solve a small TSP exactly by enumerating all permutations."""

    point_count = distance_matrix_km.shape[0]
    if distance_matrix_km.shape != (point_count, point_count):
        raise ValueError("Матрица расстояний должна быть квадратной.")
    if not 0 <= start_index < point_count:
        raise ValueError("Некорректный индекс стартового города.")
    if tariff <= 0:
        raise ValueError("Тариф должен быть положительным числом.")

    other_indices = [index for index in range(point_count) if index != start_index]
    best_route: list[int] | None = None
    best_distance = float("inf")

    for permutation in permutations(other_indices):
        route = [start_index, *permutation, start_index]
        distance = sum(
            float(distance_matrix_km[route[index], route[index + 1]])
            for index in range(len(route) - 1)
        )
        if distance < best_distance:
            best_distance = distance
            best_route = route

    if best_route is None:
        raise ValueError("Не удалось построить маршрут.")

    return TspResult(
        route_indices=best_route,
        total_distance_km=best_distance,
        total_cost=best_distance * tariff,
    )
