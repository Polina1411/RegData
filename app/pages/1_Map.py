import json
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from regdata_core.data_processing.wdi import fetch_wdi, list_countries
from regdata_core.data_processing.cache import (
    WDI_PATH, COUNTRIES_PATH, save_parquet, load_parquet
)

st.set_page_config(page_title="RegData — Map", layout="wide")
st.title("🌍 RegData: choropleth (WDI) → клик по стране → графики (без маркеров)")

# ---------- Paths ----------
GEOJSON_PATH = Path("data/raw/countries.geojson")
ISO_SOURCE_KEY = "ISO3166-1-Alpha-3"  # твой ключ ISO3 в GeoJSON properties

# ---------- Download / cache controls ----------
st.subheader("Данные (WDI / World Bank)")
colA, colB = st.columns([1, 1])

with colA:
    if st.button("⬇️ Скачать/обновить WDI (2000–2024)"):
        try:
            with st.spinner("Качаю WDI (10–60 секунд)..."):
                wdi = fetch_wdi(start_year=2000, end_year=2024)
                countries = list_countries()
            save_parquet(wdi, WDI_PATH)
            save_parquet(countries, COUNTRIES_PATH)
            st.success("✅ Готово: данные сохранены в data/processed/")
            st.write("WDI rows:", len(wdi), "| Countries rows:", len(countries))
        except Exception as e:
            st.error("❌ Не получилось скачать/сохранить WDI.")
            st.exception(e)

with colB:
    st.write("Кэш-файлы:")
    st.code(f"{WDI_PATH}\n{COUNTRIES_PATH}")

st.caption("Клик по стране — прямо по границам (GeoJSON). Выбор сохраняется.")
st.divider()

# ---------- UI controls ----------
metric = st.selectbox(
    "Показатель (WDI)",
    ["gdp_pc_usd", "inflation_cpi", "unemployment"],
    index=0
)
year = st.slider("Год (для окраски карты)", 2000, 2024, 2019)

# ---------- Cached loaders ----------
@st.cache_data
def load_wdi_cached(path_str: str) -> pd.DataFrame:
    return load_parquet(Path(path_str))

@st.cache_data
def load_geojson_cached(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Preconditions ----------
if not WDI_PATH.exists():
    st.warning("WDI кэш не найден. Нажми кнопку «Скачать/обновить WDI» выше.")
    st.stop()

if not GEOJSON_PATH.exists():
    st.error("Не найден файл границ стран: data/raw/countries.geojson")
    st.info(
        "Скачай его командой:\n\n"
        "curl -L -o data/raw/countries.geojson "
        "https://raw.githubusercontent.com/datasets/geo-countries/main/data/countries.geojson"
    )
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH))
geojson = load_geojson_cached(str(GEOJSON_PATH))

# ---------- Normalize GeoJSON: create properties['iso3'] ----------
for feat in geojson.get("features", []):
    props = feat.get("properties", {})
    iso3 = props.get(ISO_SOURCE_KEY)
    if iso3:
        props["iso3"] = iso3

# ---------- Prepare choropleth data for selected year ----------
df_year = wdi[wdi["year"] == year][["iso3", metric]].copy()
df_year = df_year.dropna(subset=[metric])

data_for_map = df_year.rename(columns={"iso3": "iso3", metric: "value"})

# ---------- session_state ----------
if "selected_iso3" not in st.session_state:
    st.session_state.selected_iso3 = None

# ---------- Map ----------
m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")

# 1) Choropleth layer (раскраска)
folium.Choropleth(
    geo_data=geojson,
    data=data_for_map,
    columns=["iso3", "value"],
    key_on="feature.properties.iso3",
    fill_opacity=0.75,
    line_opacity=0.25,
    nan_fill_opacity=0.15,
    legend_name=f"{metric} ({year})",
).add_to(m)

# 2) Clickable overlay layer поверх choropleth:
#    - fillOpacity=0, чтобы не “перекрашивать”
#    - popup содержит только iso3 (чтобы st_folium возвращал чистое значение)
def _overlay_style(_feature):
    return {"fillOpacity": 0.0, "color": "#000000", "weight": 0.35}

overlay = folium.GeoJson(
    geojson,
    name="clickable_overlay",
    style_function=_overlay_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["name", "iso3"],
        aliases=["Country", "ISO3"],
        sticky=False,
    ),
    popup=folium.GeoJsonPopup(
        fields=["iso3"],
        labels=False,
        localize=False,
    ),
)
overlay.add_to(m)

# Render
out = st_folium(m, width=None, height=600)

# ---------- Click handling ----------
# Для GeoJsonPopup st_folium обычно возвращает значение в last_object_clicked_popup
clicked_iso3 = None
if out and out.get("last_object_clicked_popup"):
    val = out["last_object_clicked_popup"]
    # иногда может прийти строка с пробелами/переводами — подчистим
    if isinstance(val, str):
        clicked_iso3 = val.strip()
    else:
        clicked_iso3 = str(val).strip()

if clicked_iso3:
    st.session_state.selected_iso3 = clicked_iso3

selected_iso3 = st.session_state.selected_iso3

# ---------- Layout ----------
left, right = st.columns([1, 2])

with left:
    st.subheader("Выбор")
    if selected_iso3:
        st.success(f"Выбрана страна: {selected_iso3}")
        if st.button("Сбросить выбор"):
            st.session_state.selected_iso3 = None
            st.rerun()
    else:
        st.info("Кликни по стране на карте (по области страны).")

    st.divider()
    st.subheader("Подсказка")
    st.write(
        "• Цвет = значение показателя за выбранный год.\n"
        "• Клик по стране идёт по GeoJSON-границам.\n"
        "• Справа — временной ряд по всем годам."
    )

with right:
    st.subheader("График по стране (реальные данные WDI)")
    if not selected_iso3:
        st.write("Выбери страну кликом по карте.")
    else:
        ts = wdi[wdi["iso3"] == selected_iso3].sort_values("year")
        if ts.empty or metric not in ts.columns:
            st.warning("Нет данных для этой страны/показателя.")
        else:
            st.line_chart(ts.set_index("year")[metric])
            st.caption("Последние значения:")
            st.dataframe(ts[["iso3", "year", metric]].tail(10), use_container_width=True)
