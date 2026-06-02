from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import build_analytics_dataset, build_country_lookup, country_label, country_options, current_year, metrics_for_df
from regdata_core.data_processing.cache import WDI_PATH, COUNTRIES_PATH, OECD_RECENT_PATH, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Временные ряды", layout="wide")
apply_app_style()
CURRENT_YEAR = current_year()

render_hero(
    "Временные ряды",
    "Сравнивай динамику одного показателя сразу по нескольким странам и смотри, как расходятся их долгосрочные траектории.",
)

@st.cache_data
def load_wdi_cached(path_str: str) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


if not WDI_PATH.exists():
    st.warning("Данные WDI не найдены. Сначала открой страницу карты и обнови данные.")
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
METRICS = metrics_for_df(wdi)
countries = load_countries_cached(str(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)
iso_options = country_options(wdi, country_lookup)

default_iso3 = st.session_state.get("selected_iso3")
default_selection = [default_iso3] if default_iso3 in iso_options else iso_options[:3]

col1, col2 = st.columns([1.1, 1.9])
with col1:
    metric = st.selectbox("Показатель", METRICS, format_func=metric_label)
with col2:
    selected_iso3 = st.multiselect(
        "Страны",
        iso_options,
        default=default_selection,
        format_func=lambda iso: country_label(iso, country_lookup),
    )

available_years = wdi["year"].dropna().astype(int)
min_year = int(available_years.min())
max_year = int(available_years.max())
slider_max_year = max(CURRENT_YEAR, max_year)
year_from, year_to = st.slider("Период", min_year, slider_max_year, (2010, slider_max_year))

if slider_max_year > max_year:
    render_note(
        f"Годы {max_year + 1}-{slider_max_year} доступны в временном диапазоне, "
        "но значения за эти годы могут быть ещё не опубликованы."
    )

if metric in {"inflation_cpi", "unemployment"} and OECD_RECENT_PATH.exists():
    render_note(
        "Для 2025–2026 используются реальные месячные наблюдения OECD, приведённые к последнему доступному месяцу года."
    )

if not selected_iso3:
    st.info("Выбери хотя бы одну страну.")
    st.stop()

plot_df = wdi[
    (wdi["iso3"].isin(selected_iso3)) &
    (wdi["year"].between(year_from, year_to))
][["iso3", "year", metric]].copy()

plot_df["Страна"] = plot_df["iso3"].map(lambda iso: country_label(iso, country_lookup))

chart_df = plot_df.dropna(subset=[metric])
if chart_df.empty:
    st.warning("Для выбранного периода нет данных.")
    st.stop()

# В этом разделе сравниваем один показатель между странами, поэтому оставляем одну общую ось Y.
st.subheader(f"Динамика: {metric_label(metric)}")
render_note(
    "Каждая линия — это страна. Точки на линии показывают отдельные годы, поэтому можно увидеть не только тренд, но и плотность наблюдений."
)
fig = px.line(
    chart_df,
    x="year",
    y=metric,
    color="Страна",
    markers=True,
    color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5"],
)
fig.update_layout(
    xaxis_title="Год",
    yaxis_title=metric_label(metric),
    plot_bgcolor="rgba(255,255,255,0.7)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_traces(
    line=dict(width=3.4, shape="spline", smoothing=0.55),
    marker=dict(size=7, line=dict(color="white", width=1)),
)
st.plotly_chart(fig, use_container_width=True)

latest_snapshot = (
    # Быстрый срез на конец периода нужен, чтобы не вычитывать последнее значение глазами с графика.
    plot_df[plot_df["year"] == year_to][["Страна", metric]]
    .rename(columns={metric: metric_label(metric)})
    .sort_values("Страна")
    .fillna("—")
)

left, right = st.columns([1, 1])
with left:
    st.subheader(f"Срез на {year_to} год")
    st.dataframe(latest_snapshot, use_container_width=True)
with right:
    st.subheader("Данные по всему периоду")
    st.dataframe(
        plot_df[["Страна", "year", metric]]
        .rename(columns={"year": "Год", metric: metric_label(metric)})
        .fillna("—"),
        use_container_width=True,
    )
