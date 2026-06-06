import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.analytics.convergence import prepare_convergence_matrix, run_kmeans, summarize_clubs
from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    current_year,
    metrics_for_df,
)
from regdata_core.data_processing.cache import (
    COUNTRIES_PATH,
    LIGHT_COUNTRIES_GEOJSON_PATH,
    OECD_RECENT_PATH,
    RAW_COUNTRIES_GEOJSON_PATH,
    WDI_PATH,
    file_version,
    load_parquet,
)
from regdata_core.data_processing.geojson import build_lightweight_geojson
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Клубы конвергенции", layout="wide")
apply_app_style()

render_hero(
    "Клубы конвергенции",
    "Простая кластеризация стран по форме одного временного ряда: "
    "страны с похожей динамикой попадают в один клуб.",
)


@st.cache_data
def load_wdi_cached(path_str: str, _version: int) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


@st.cache_data
def load_geojson_cached(path_str: str, _version: int) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_lightweight_geojson() -> Path | None:
    if LIGHT_COUNTRIES_GEOJSON_PATH.exists():
        return LIGHT_COUNTRIES_GEOJSON_PATH
    if not RAW_COUNTRIES_GEOJSON_PATH.exists():
        return None
    build_lightweight_geojson(
        source_path=RAW_COUNTRIES_GEOJSON_PATH,
        output_path=LIGHT_COUNTRIES_GEOJSON_PATH,
        iso_key="ISO3166-1-Alpha-3",
        decimals=1,
    )
    return LIGHT_COUNTRIES_GEOJSON_PATH


def build_club_name(club_id: int) -> str:
    return f"Клуб {club_id}"


def short_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "Недостаточно данных для краткого вывода."

    latest = summary_df.sort_values("year").groupby("club", as_index=False).tail(1)
    highest = latest.sort_values("mean_value", ascending=False).iloc[0]["club"]
    lowest = latest.sort_values("mean_value", ascending=True).iloc[0]["club"]
    return (
        f"На конце периода выше остальных расположен {build_club_name(int(highest))}, "
        f"а ниже остальных — {build_club_name(int(lowest))}. "
        "Смысл графика в том, чтобы сравнивать именно форму траекторий."
    )


if not WDI_PATH.exists():
    st.warning("Данные WDI не найдены. Сначала обнови данные на странице карты.")
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH), file_version(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)
metrics = metrics_for_df(wdi)

year_values = wdi["year"].dropna().astype(int)
min_year = int(year_values.min())
max_year = int(year_values.max()) if not year_values.empty else current_year()

col1, col2, col3 = st.columns([1.2, 1.5, 1])
with col1:
    metric = st.selectbox("Показатель", metrics, format_func=metric_label)
with col2:
    year_from, year_to = st.slider("Период анализа", min_year, max_year, (max(min_year, 2010), max_year))
with col3:
    n_clusters = st.slider("Количество клубов", 2, 6, 3)

coverage = st.slider("Минимальная полнота ряда", 0.4, 1.0, 0.7, 0.05)

matrix_df, years = prepare_convergence_matrix(
    wdi,
    metric=metric,
    year_from=year_from,
    year_to=year_to,
    min_coverage=coverage,
)

if matrix_df.empty:
    st.warning("Недостаточно данных для построения клубов конвергенции в выбранном диапазоне.")
    st.stop()

clustered_df = run_kmeans(matrix_df, n_clusters=n_clusters)
summary_df, members_df = summarize_clubs(clustered_df, years)
members_df["Страна"] = members_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
members_df["Клуб"] = members_df["club"].map(build_club_name)
summary_df["Клуб"] = summary_df["club"].map(build_club_name)

render_note(
    "Алгоритм группирует страны по форме временного ряда. "
    "Если у стран похожий рост, спад или стабильность, они попадают в один клуб."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Стран в анализе", int(len(clustered_df)))
with c2:
    st.metric("Клубов", int(clustered_df["club"].nunique()))
with c3:
    st.metric("Период", f"{year_from}–{year_to}")

st.subheader("Средние траектории клубов")
fig = px.line(
    summary_df,
    x="year",
    y="mean_value",
    color="Клуб",
    markers=True,
    color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
)
fig.update_layout(
    xaxis_title="Год",
    yaxis_title=metric_label(metric),
    plot_bgcolor="rgba(255,255,255,0.74)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)
render_note(short_summary(summary_df))

left, right = st.columns([1, 1])
with left:
    st.subheader("Размеры клубов")
    club_sizes = members_df.groupby("Клуб", as_index=False).size().rename(columns={"size": "Стран"})
    fig_sizes = px.bar(
        club_sizes,
        x="Клуб",
        y="Стран",
        color="Клуб",
        color_discrete_sequence=["#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9", "#b7b7a4"],
    )
    fig_sizes.update_layout(
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

with right:
    st.subheader("Участники клубов")
    st.dataframe(
        members_df[["Клуб", "Страна"]].sort_values(["Клуб", "Страна"]),
        use_container_width=True,
        hide_index=True,
    )

geojson_path = ensure_lightweight_geojson()
if geojson_path is None:
    render_note("Файл геометрии стран не найден, поэтому карта клубов сейчас недоступна.")
else:
    st.subheader("Карта клубов")
    geojson = load_geojson_cached(str(geojson_path), file_version(geojson_path))
    map_df = members_df[["iso3", "club", "Клуб", "Страна"]].copy()

    fig_map = px.choropleth(
        map_df,
        geojson=geojson,
        featureidkey="properties.iso3",
        locations="iso3",
        color="Клуб",
        hover_name="Страна",
        projection="natural earth",
        color_discrete_sequence=["#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9", "#b7b7a4"],
    )
    fig_map.update_geos(
        showcountries=True,
        countrycolor="rgba(255,255,255,0.55)",
        showcoastlines=False,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig_map.update_layout(
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)
