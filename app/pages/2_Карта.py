import json
from datetime import date
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
import plotly.express as px
from streamlit_folium import st_folium

from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    metrics_for_df,
    prepare_hover_analytics,
)
from regdata_core.data_processing.wdi import fetch_wdi, list_countries
from regdata_core.data_processing.efi import fetch_efi_official, save_efi_parquet
from regdata_core.data_processing.oecd import fetch_oecd_recent
from regdata_core.data_processing.geojson import build_lightweight_geojson
from regdata_core.data_processing.cache import (
    WDI_PATH,
    COUNTRIES_PATH,
    OECD_RECENT_PATH,
    EFI_PATH,
    RAW_COUNTRIES_GEOJSON_PATH,
    LIGHT_COUNTRIES_GEOJSON_PATH,
    save_parquet,
    load_parquet,
)
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label

st.set_page_config(page_title="RegData — Карта", layout="wide")
apply_app_style()
CURRENT_YEAR = date.today().year
render_hero(
    "Карта стран",
    "Сравнивай страны по ключевым показателям и переходи от карты к динамике выбранной страны одним нажатием.",
)

ISO_SOURCE_KEY = "ISO3166-1-Alpha-3"

st.subheader("Источник данных")
colA, colB = st.columns([1, 1])

with colA:
    if st.button(f"Обновить данные WDI за 2000-{CURRENT_YEAR}"):
        try:
            with st.spinner("Загрузка данных World Bank..."):
                wdi = fetch_wdi(start_year=2000, end_year=CURRENT_YEAR)
                countries = list_countries()
            save_parquet(wdi, WDI_PATH)
            save_parquet(countries, COUNTRIES_PATH)
            st.success("Данные обновлены")
            st.write("Строк WDI:", len(wdi), "Стран:", len(countries))
        except Exception as e:
            st.error("Не удалось загрузить данные WDI")
            st.exception(e)

    if st.button("Обновить OECD для 2025-2026"):
        try:
            with st.spinner("Загрузка реальных данных OECD..."):
                oecd_recent = fetch_oecd_recent()
            save_parquet(oecd_recent, OECD_RECENT_PATH)
            st.success("Данные OECD обновлены")
            st.write("Строк OECD:", len(oecd_recent))
        except Exception as e:
            st.error("Не удалось загрузить данные OECD")
            st.exception(e)

    if st.button("Обновить индекс экономической свободы"):
        try:
            with st.spinner("Загрузка индекса экономической свободы..."):
                efi_df = fetch_efi_official()
            save_efi_parquet(efi_df, EFI_PATH)
            st.success("Индекс экономической свободы загружен")
            st.write("Строк EFI:", len(efi_df))
        except Exception as e:
            st.error("Не удалось загрузить индекс экономической свободы")
            st.exception(e)

with colB:
    render_note(
        "Источник: World Development Indicators для базовых рядов и OECD для реальных "
        "значений инфляции и безработицы за 2025–2026."
    )
    if EFI_PATH.exists():
        render_note("Индекс экономической свободы подключён автоматически и доступен как отдельный показатель.")

@st.cache_data
def load_wdi_cached(path_str: str) -> pd.DataFrame:
    return load_parquet(Path(path_str))

@st.cache_data
def load_geojson_cached(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def prepare_year_slice(wdi: pd.DataFrame, year: int, metric: str) -> pd.DataFrame:
    # Для карты заранее режем данные до одного года и одного показателя — так отклик заметно быстрее.
    return (
        wdi.loc[wdi["year"] == year, ["iso3", metric]]
        .dropna()
        .rename(columns={metric: "value"})
    )


def build_geojson_for_metric(geojson: dict, map_df: pd.DataFrame) -> dict:
    # В hover добавляем уже подготовленные подписи, чтобы не собирать их на лету внутри folium.
    features = []
    hover_lookup = map_df.set_index("iso3").to_dict(orient="index") if not map_df.empty else {}

    for feature in geojson.get("features", []):
        properties = dict(feature.get("properties", {}))
        iso3 = properties.get("iso3")
        hover = hover_lookup.get(iso3, {})
        properties["value_text"] = hover.get("value_text", "—")
        properties["rank_text"] = f"#{int(hover['rank'])}" if pd.notna(hover.get("rank")) else "—"
        properties["hover_category"] = hover.get("hover_category", "—")

        features.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": properties,
            }
        )

    return {"type": "FeatureCollection", "features": features}

def ensure_lightweight_geojson() -> Path:
    if LIGHT_COUNTRIES_GEOJSON_PATH.exists():
        return LIGHT_COUNTRIES_GEOJSON_PATH

    build_lightweight_geojson(
        source_path=RAW_COUNTRIES_GEOJSON_PATH,
        output_path=LIGHT_COUNTRIES_GEOJSON_PATH,
        iso_key=ISO_SOURCE_KEY,
        decimals=1,
    )
    return LIGHT_COUNTRIES_GEOJSON_PATH

if not WDI_PATH.exists():
    st.warning("Файл с кэшированными данными WDI не найден")
    st.stop()

if not RAW_COUNTRIES_GEOJSON_PATH.exists():
    st.error("Файл геометрии стран не найден")
    st.stop()

with st.spinner("Подготовка карты..."):
    geojson_path = ensure_lightweight_geojson()

wdi = load_wdi_cached(str(WDI_PATH))
oecd_recent = load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame()
wdi = build_analytics_dataset(wdi, oecd_recent)
countries = load_parquet(COUNTRIES_PATH) if COUNTRIES_PATH.exists() else pd.DataFrame(columns=["iso3", "country"])
country_lookup = build_country_lookup(countries)
geojson = load_geojson_cached(str(geojson_path))

available_years = wdi["year"].dropna().astype(int)
max_available_year = int(available_years.max()) if not available_years.empty else CURRENT_YEAR
slider_max_year = max(CURRENT_YEAR, max_available_year)
default_year = 2019 if 2000 <= 2019 <= slider_max_year else slider_max_year

metric = st.selectbox(
    "Показатель",
    metrics_for_df(wdi),
    format_func=metric_label,
    index=0
)
year = st.slider("Год", 2000, slider_max_year, default_year)

if slider_max_year > max_available_year:
    render_note(
        f"Годы {max_available_year + 1}-{slider_max_year} добавлены в таблицы и ряды, "
        "но значения World Bank за эти годы могут быть ещё не опубликованы."
    )

if metric in {"inflation_cpi", "unemployment"} and OECD_RECENT_PATH.exists():
    render_note(
        "Для 2025–2026 по этому показателю используются реальные последние месячные наблюдения OECD внутри соответствующего года."
    )

data_for_map = prepare_year_slice(wdi, year, metric)
hover_map_df = prepare_hover_analytics(data_for_map, metric)
map_geojson = build_geojson_for_metric(geojson, hover_map_df)

if "selected_iso3" not in st.session_state:
    st.session_state.selected_iso3 = None

m = folium.Map(location=[20, 0], zoom_start=2, tiles=None, prefer_canvas=True)

m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-attribution a {
    display: none !important;
}
</style>
"""))


choropleth = folium.Choropleth(
    # Карта здесь — главный вход в данные, поэтому оставляем её максимально “чистой” и без лишнего шума.
    geo_data=map_geojson,
    data=data_for_map,
    columns=["iso3", "value"],
    key_on="feature.properties.iso3",
    fill_opacity=0.75,
    line_opacity=0.25,
    nan_fill_opacity=0.15,
    legend_name=f"{metric_label(metric)} ({year})",
    smooth_factor=1.5,
).add_to(m)

choropleth.geojson.style_function = lambda _feature: {
    "fillOpacity": 0.75,
    "color": "#f8f4ec",
    "weight": 0.45,
}
choropleth.geojson.highlight_function = lambda _feature: {
    "fillOpacity": 0.92,
    "color": "#19332d",
    "weight": 1.5,
}

choropleth.geojson.add_child(
    folium.GeoJsonTooltip(
        fields=["name", "iso3", "value_text", "rank_text", "hover_category"],
        aliases=["Страна", "ISO3", metric_label(metric), "Место", "Категория"],
        sticky=True,
        labels=True,
        style=(
            "background-color: rgba(250,248,242,0.96);"
            "border: 1px solid rgba(31,58,51,0.18);"
            "border-radius: 12px;"
            "box-shadow: 0 14px 30px rgba(31,58,51,0.12);"
            "padding: 10px 12px;"
            "font-size: 13px;"
        ),
    )
)
choropleth.geojson.add_child(
    folium.GeoJsonPopup(
        fields=["iso3"],
        labels=False,
        localize=False,
    )
)

out = st_folium(
    m,
    width=None,
    height=600,
    returned_objects=["last_object_clicked_popup"],
)

clicked_iso3 = None
if out and out.get("last_object_clicked_popup"):
    val = out["last_object_clicked_popup"]
    clicked_iso3 = str(val).strip()

if clicked_iso3:
    st.session_state.selected_iso3 = clicked_iso3

selected_iso3 = st.session_state.selected_iso3

left, right = st.columns([1, 2])

with left:
    st.subheader("Выбор")
    if selected_iso3:
        st.success(country_label(selected_iso3, country_lookup))
        if st.button("Сбросить"):
            st.session_state.selected_iso3 = None
            st.rerun()
    else:
        st.info("Нажми на страну на карте")

with right:
    st.subheader("Динамика по годам")
    if not selected_iso3:
        st.write("Выбери страну на карте")
    else:
        ts = wdi[wdi["iso3"] == selected_iso3].sort_values("year")
        if ts.empty or metric not in ts.columns:
            st.warning("Нет данных для выбранной страны")
        else:
            st.caption(country_label(selected_iso3, country_lookup))
            chart_ts = ts[["year", metric]].dropna().copy()
            fig_ts = px.line(
                chart_ts,
                x="year",
                y=metric,
                markers=True,
                color_discrete_sequence=["#2f5d50"],
            )
            fig_ts.update_layout(
                xaxis_title="Год",
                yaxis_title=metric_label(metric),
                plot_bgcolor="rgba(255,255,255,0.72)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                hovermode="x unified",
            )
            fig_ts.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
            fig_ts.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
            fig_ts.update_traces(
                line=dict(width=3.2, shape="spline", smoothing=0.55),
                marker=dict(size=7, line=dict(color="white", width=1)),
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            table_df = ts[["iso3", "year", metric]].tail(10).copy()
            table_df["Страна"] = table_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
            st.dataframe(
                table_df.rename(
                    columns={
                        "iso3": "ISO3",
                        "year": "Год",
                        metric: metric_label(metric),
                    }
                )[["Страна", "ISO3", "Год", metric_label(metric)]].fillna("—"),
                use_container_width=True
            )
