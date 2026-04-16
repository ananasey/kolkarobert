"""Streamlit application for urban map analysis and route optimization."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.building_analysis import analyze_building_density
from analysis.green_analysis import analyze_green_areas
from analysis.transport_analysis import analyze_transport_infrastructure
from config import DEFAULT_ZOOM, get_settings
from routing.road_distance import DistanceMatrixError, calculate_distance_matrix
from routing.route_visualization import render_route_graph
from routing.tsp_solver import solve_tsp_exact
from services.geocoder import GeoPoint, GeocodingError, geocode_place
from services.static_maps import StaticMapError, load_static_map
from utils.helpers import has_duplicates, parse_places


st.set_page_config(
    page_title="Городская среда и оптимальный маршрут",
    page_icon="🗺️",
    layout="wide",
)


def show_api_key_warning() -> None:
    """Show a visible warning when Yandex API key is not configured."""

    settings = get_settings()
    missing = []
    if not settings.has_geocoder_key:
        missing.append("геокодер")
    if not settings.has_distance_matrix_key:
        missing.append("Distance Matrix")

    if missing:
        st.warning(
            "В .env не хватает ключей для сервисов: "
            f"{', '.join(missing)}. Можно указать отдельные ключи или один общий YANDEX_API_KEY."
        )


def show_geocoding_candidates(candidates: list[GeoPoint]) -> None:
    """Display a warning when geocoder returned several candidates."""

    if len(candidates) <= 1:
        return
    with st.expander("Геокодер вернул несколько вариантов"):
        st.write("Приложение использует первый вариант, но показывает остальные для контроля.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Название": item.name,
                        "Адрес": item.address,
                        "Широта": item.lat,
                        "Долгота": item.lon,
                        "Точность": item.precision,
                    }
                    for item in candidates
                ]
            ),
            use_container_width=True,
        )


def show_image(image, *, caption: str | None = None) -> None:
    """Display an image with compatibility across Streamlit versions."""

    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def render_urban_analysis_tab() -> None:
    """Render Streamlit UI for task 1."""

    st.header("Анализ городской среды")
    st.write(
        "Введите город, получите изображение Яндекс.Карт и оцените его "
        "методами компьютерного зрения."
    )

    city = st.text_input("Город для анализа", value="Екатеринбург")
    analysis_type = st.selectbox(
        "Тип анализа",
        (
            "Зелёные зоны",
            "Плотность/тип застройки",
            "Транспортная инфраструктура",
        ),
    )
    zoom = st.slider("Масштаб карты", min_value=11, max_value=16, value=DEFAULT_ZOOM)

    if not st.button("Загрузить карту и выполнить анализ", type="primary"):
        return

    settings = get_settings()
    try:
        candidates = geocode_place(city, settings, results=5)
        center = candidates[0]
        show_geocoding_candidates(candidates)
        layer = (
            "map"
            if analysis_type == "Транспортная инфраструктура"
            else "sat"
        )
        image = load_static_map(center, settings, zoom=zoom, layer=layer)
    except (GeocodingError, StaticMapError) as exc:
        st.error(str(exc))
        return

    st.success(
        f"Изображение загружено для точки: "
        f"{center.name} ({center.lat:.5f}, {center.lon:.5f})"
    )

    if analysis_type == "Зелёные зоны":
        result = analyze_green_areas(image, latitude=center.lat, zoom=zoom)
        metric_title = "Процент озеленения"
        metric_value = f"{result.green_percent:.2f}%"
        conclusion = result.conclusion
        mask_image = result.mask_image
        overlay_image = result.overlay_image
        extra_metrics = [
            ("Площадь зелёных зон", f"{result.green_area_km2:.2f} км²"),
            ("Площадь снимка", f"{result.total_area_km2:.2f} км²"),
        ]
    elif analysis_type == "Плотность/тип застройки":
        result = analyze_building_density(image)
        metric_title = f"Плотность застройки: {result.density_class}"
        metric_value = f"{result.density_index:.2f}"
        conclusion = result.conclusion
        mask_image = result.mask_image
        overlay_image = result.overlay_image
        extra_metrics = [
            ("Застроенная территория", f"{result.built_percent:.2f}%"),
            ("Границы/текстура", f"{result.edge_density_percent:.2f}%"),
            ("Компоненты", str(result.contour_count)),
        ]
    else:
        result = analyze_transport_infrastructure(image)
        metric_title = "Индекс транспортной развитости"
        metric_value = f"{result.transport_index:.2f}"
        conclusion = result.conclusion
        mask_image = result.mask_image
        overlay_image = result.overlay_image
        extra_metrics = [
            ("Доля дорожной сети", f"{result.transport_percent:.2f}%"),
            ("Длинные линии", str(result.long_line_count)),
        ]

    metric_columns = st.columns(1 + len(extra_metrics))
    metric_columns[0].metric(metric_title, metric_value)
    for column, (title, value) in zip(metric_columns[1:], extra_metrics):
        column.metric(title, value)
    st.info(conclusion)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Исходное изображение")
        show_image(image)
    with col2:
        st.subheader("Маска")
        show_image(mask_image)
    with col3:
        st.subheader("Итоговая визуализация")
        show_image(overlay_image)


def _build_places_table(points: list[GeoPoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Населённый пункт": point.name,
                "Адрес геокодера": point.address,
                "Широта": round(point.lat, 6),
                "Долгота": round(point.lon, 6),
                "Точность": point.precision,
            }
            for point in points
        ]
    )


def _matrix_dataframe(matrix, names: list[str], decimals: int = 2) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=names, columns=names).round(decimals)


def render_route_tab() -> None:
    """Render Streamlit UI for task 2."""

    st.header("Оптимальный маршрут")
    st.write(
        "Введите ровно 7 населённых пунктов РФ. Приложение геокодирует их, "
        "строит матрицу расстояний и решает задачу коммивояжёра точным перебором."
    )

    default_places = "\n".join(
        ["Екатеринбург", "Пермь", "Тюмень", "Челябинск", "Курган", "Уфа", "Ижевск"]
    )
    raw_places = st.text_area(
        "7 населённых пунктов, каждый с новой строки",
        value=default_places,
        height=190,
    )
    tariff = st.number_input(
        "Тариф доставки за 1 км",
        min_value=0.01,
        value=45.0,
        step=1.0,
        format="%.2f",
    )

    places = parse_places(raw_places)
    if len(places) == 7 and not has_duplicates(places):
        start_city = st.selectbox("Стартовый город", places)
    else:
        start_city = None
        st.selectbox("Стартовый город", ["Сначала введите ровно 7 уникальных пунктов"], disabled=True)

    if not st.button("Рассчитать оптимальный маршрут", type="primary"):
        return

    if len(places) != 7:
        st.error(f"Нужно ввести ровно 7 населённых пунктов. Сейчас введено: {len(places)}.")
        return
    if has_duplicates(places):
        st.error("В списке есть дублирующиеся населённые пункты. Уберите повторы.")
        return
    if start_city is None:
        st.error("Не выбран стартовый город.")
        return
    if tariff <= 0:
        st.error("Тариф должен быть положительным числом.")
        return

    settings = get_settings()
    points: list[GeoPoint] = []
    ambiguous_rows: list[dict[str, str]] = []

    try:
        for place in places:
            candidates = geocode_place(place, settings, results=3)
            chosen = candidates[0]
            points.append(chosen)
            if len(candidates) > 1:
                ambiguous_rows.append(
                    {
                        "Запрос": place,
                        "Использован": chosen.address or chosen.name,
                        "Другие варианты": "; ".join(
                            candidate.address or candidate.name for candidate in candidates[1:]
                        ),
                    }
                )
    except GeocodingError as exc:
        st.error(str(exc))
        return

    if ambiguous_rows:
        with st.expander("Неоднозначные результаты геокодирования"):
            st.dataframe(pd.DataFrame(ambiguous_rows), use_container_width=True)

    try:
        distance_result = calculate_distance_matrix(points, settings)
    except DistanceMatrixError as exc:
        st.error(str(exc))
        return

    start_index = places.index(start_city)
    try:
        tsp_result = solve_tsp_exact(distance_result.matrix_km, start_index, tariff)
    except ValueError as exc:
        st.error(str(exc))
        return

    names = [point.name for point in points]
    route_names = [names[index] for index in tsp_result.route_indices]
    cost_matrix = distance_result.matrix_km * tariff

    if distance_result.used_fallback:
        st.warning(distance_result.message)
    else:
        st.success(distance_result.message)

    st.subheader("Населённые пункты и координаты")
    st.dataframe(_build_places_table(points), use_container_width=True)

    st.subheader("Матрица расстояний, км")
    st.dataframe(_matrix_dataframe(distance_result.matrix_km, names), use_container_width=True)

    with st.expander("Режим расстояний для каждой пары"):
        st.dataframe(pd.DataFrame(distance_result.mode_matrix, index=names, columns=names))

    st.subheader("Матрица стоимости")
    st.dataframe(_matrix_dataframe(cost_matrix, names), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Длина маршрута", f"{tsp_result.total_distance_km:,.2f} км")
    col2.metric("Стоимость маршрута", f"{tsp_result.total_cost:,.2f}")

    st.subheader("Оптимальный маршрут")
    st.write(" → ".join(route_names))
    show_image(render_route_graph(points, tsp_result.route_indices))


def main() -> None:
    """Application entry point."""

    st.title("Городская среда и оптимальный маршрут")
    show_api_key_warning()

    tab_urban, tab_route = st.tabs(["Анализ городской среды", "Оптимальный маршрут"])
    with tab_urban:
        render_urban_analysis_tab()
    with tab_route:
        render_route_tab()


if __name__ == "__main__":
    main()
