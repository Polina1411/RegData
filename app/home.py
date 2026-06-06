from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import build_analytics_dataset, build_country_lookup, country_label, metrics_for_df
from regdata_core.data_processing.cache import COUNTRIES_PATH, OECD_RECENT_PATH, WDI_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, metric_label, render_hero, render_note, render_panel


st.set_page_config(page_title="RegData — Главная", layout="wide")
apply_app_style()

CURRENT_YEAR = date.today().year


@st.cache_data
def load_wdi_cached(path_str: str, _version: int) -> pd.DataFrame:
    # Главная страница часто перерисовывается, поэтому данные кэшируем сразу.
    return load_parquet(Path(path_str))


@st.cache_data
def load_optional_parquet(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return load_parquet(path)


@st.cache_data
def load_countries_cached(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)

render_hero(
    "RegData",
    "Минималистичная аналитическая платформа для чтения мировой экономики: "
    "карта, сравнение стран, временные ряды и экономические связи в одном визуальном ритме.",
)

hero_left, hero_right = st.columns([1.05, 1.95])

with hero_left:
    render_panel(
        "Платформа",
        "Стильный слой над данными",
        "RegData собирает реальные международные данные и превращает их в понятный, аккуратный и современный аналитический интерфейс.",
    )
    st.markdown(" ")
    if WDI_PATH.exists():
        # На главной показываем уже собранный аналитический датасет, а не сырые куски по отдельности.
        wdi = load_wdi_cached(str(WDI_PATH), file_version(WDI_PATH))
        wdi = build_analytics_dataset(
            wdi,
            load_optional_parquet(str(OECD_RECENT_PATH), file_version(OECD_RECENT_PATH)),
        )
        countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
        country_lookup = build_country_lookup(countries)
        metrics = metrics_for_df(wdi)
        available_years = wdi["year"].dropna().astype(int)
        max_year = int(available_years.max()) if not available_years.empty else CURRENT_YEAR
        slider_max_year = max(CURRENT_YEAR, max_year)
        year = st.slider("Год фокуса", 2000, slider_max_year, min(2024, slider_max_year))
        metric = st.selectbox("Показатель", metrics, format_func=metric_label)
    else:
        year = st.slider("Год фокуса", 2000, CURRENT_YEAR, min(2024, CURRENT_YEAR))
        metric = "gdp_pc_usd"
        wdi = pd.DataFrame()
        country_lookup = {}

with hero_right:
    stat_1, stat_2, stat_3 = st.columns(3)
    with stat_1:
        render_panel("Источник", "World Bank", "Основные макроэкономические ряды и сопоставимая статистика по странам.")
    with stat_2:
        render_panel("Апдейт", "OECD", "Свежие данные по инфляции и безработице за 2025–2026 годы.")
    with stat_3:
        render_panel("Сценарий", "Visual first", "Карта, профиль страны, корреляции и временные ряды собраны в одном цельном сценарии.")

    if not wdi.empty and metric in wdi.columns:
        # Для первого экрана достаточно простого  top-10 среза по странам.
        df = (
            wdi.loc[wdi["year"] == year, ["iso3", metric]]
            .dropna()
            .assign(Страна=lambda data: data["iso3"].map(lambda iso: country_label(iso, country_lookup)))
            .rename(columns={metric: "Значение"})
            [["iso3", "Страна", "Значение"]]
            .sort_values("Значение", ascending=False)
            .head(10)
            .assign(Год=year, Показатель=metric_label(metric))
        )
    else:
        df = pd.DataFrame(columns=["Страна", "Значение", "Год", "Показатель"])

st.subheader("Главный срез")

if df.empty:
    render_note("На главной пока нет локально загруженных данных. Обнови данные на странице карты, и здесь появится обзор по реальным значениям.")
else:
    # Главный график здесь намеренно простой: он должен быстро задавать контекст, а не перегружать анализом.
    fig = px.bar(
        df,
        x="Страна",
        y="Значение",
        color="Значение",
        text_auto=".1f",
        color_continuous_scale=["#d8e2dc", "#a3b18a", "#588157", "#344e41"],
    )
    fig.update_layout(
        title=f"{metric_label(metric)} — {year}",
        plot_bgcolor="rgba(255,255,255,0.78)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=56, b=0),
        title_font_size=24,
        font_color="#223a34",
        title_font_color="#17352e",
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=0.95,
        textfont_size=12,
    )
    st.plotly_chart(fig, use_container_width=True)

bottom_left, bottom_mid, bottom_right = st.columns(3)
with bottom_left:
    render_panel(
        "Раздел",
        "Обзор",
        "Быстрый глобальный срез по выбранному показателю, значениям за год и общему экономическому фону.",
    )
with bottom_mid:
    render_panel(
        "Раздел",
        "Карта и страна",
        "Плавный переход от карты мира к профилю конкретной страны без лишнего шума.",
    )
with bottom_right:
    render_panel(
        "Раздел",
        "Корреляции и ряды",
        "Сравнение показателей, динамика по годам и спокойный визуальный поиск закономерностей.",
    )
