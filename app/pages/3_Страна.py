from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    country_options,
    current_year,
    efi_category,
    format_metric_value,
    latest_year_with_value,
    metrics_for_df,
)
from regdata_core.data_processing.cache import WDI_PATH, COUNTRIES_PATH, OECD_RECENT_PATH, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Страна", layout="wide")
apply_app_style()
CURRENT_YEAR = current_year()

render_hero(
    "Профиль страны",
    "Подробный взгляд на одну страну: последние значения, динамика по годам и компактная таблица для анализа.",
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
if default_iso3 not in iso_options:
    default_iso3 = iso_options[0]

col1, col2 = st.columns([1.4, 1])
with col1:
    selected_iso3 = st.selectbox(
        "Страна",
        iso_options,
        index=iso_options.index(default_iso3),
        format_func=lambda iso: country_label(iso, country_lookup),
    )

st.session_state.selected_iso3 = selected_iso3
country_df = wdi[wdi["iso3"] == selected_iso3].sort_values("year")
country_name = country_label(selected_iso3, country_lookup)
country_years = country_df["year"].dropna().astype(int)
min_country_year = int(country_years.min()) if not country_years.empty else 2000
max_country_year = int(country_years.max()) if not country_years.empty else CURRENT_YEAR
default_focus_year = max(min(2024, max_country_year), min_country_year)

with col2:
    year_focus = st.slider("Опорный год", min_country_year, max_country_year, default_focus_year)

latest_years = {metric: latest_year_with_value(country_df, metric) for metric in METRICS}
overall_latest_year = max((year for year in latest_years.values() if year is not None), default=None)

if overall_latest_year is None:
    st.warning("Для выбранной страны пока нет наблюдений.")
    st.stop()

if CURRENT_YEAR > overall_latest_year:
    render_note(
        f"Для {country_name} последний год с опубликованными значениями сейчас: {overall_latest_year}. "
        f"Годы {overall_latest_year + 1}-{CURRENT_YEAR} могут пока оставаться пустыми."
    )

if OECD_RECENT_PATH.exists():
    render_note(
        "Для инфляции и безработицы за 2025–2026 используются реальные последние месячные наблюдения OECD внутри года."
    )

st.subheader(country_name)

# Верхние карточки теперь специально привязаны к выбранному опорному году, а не к последнему значению в ряду.
metric_columns_per_row = 3
for start_idx in range(0, len(METRICS), metric_columns_per_row):
    row_metrics = METRICS[start_idx:start_idx + metric_columns_per_row]
    metric_cols = st.columns(len(row_metrics))
    for column, metric in zip(metric_cols, row_metrics):
        latest_year = latest_years[metric]
        with column:
            if latest_year is None:
                st.metric(metric_label(metric), "—", "нет данных")
                continue

            focus_row = country_df.loc[country_df["year"] == year_focus, ["year", metric]].dropna()
            if focus_row.empty:
                st.metric(
                    metric_label(metric),
                    "—",
                    f"нет данных за {year_focus}",
                )
                continue

            value = focus_row.iloc[-1][metric]
            previous_row = (
                country_df.loc[(country_df["year"] < year_focus), ["year", metric]]
                .dropna()
                .sort_values("year")
            )
            delta_text = f"год: {year_focus}"
            if metric == "efi_total":
                delta_text = f"статус: {efi_category(value)}"
            elif not previous_row.empty:
                # Для обычных метрик подсказываем изменение к предыдущему доступному году — так карточки читаются живее.
                previous_value = previous_row.iloc[-1][metric]
                previous_year = int(previous_row.iloc[-1]["year"])
                delta_text = f"{float(value - previous_value):+.2f} к {previous_year}"

            st.metric(
                metric_label(metric),
                format_metric_value(metric, value),
                delta_text,
            )

chart_df = country_df.melt(
    id_vars=["year"],
    value_vars=METRICS,
    var_name="metric",
    value_name="value",
).dropna()
chart_df["metric"] = chart_df["metric"].map(metric_label)

# Ниже уже показываем не уровень в одной точке, а форму всей траектории страны.
st.subheader("Динамика показателей")
render_note(
    "На графике каждая линия — отдельный показатель, а точки отмечают конкретные годы. Так проще заметить переломы и пропуски в динамике."
)
fig = px.line(
    chart_df,
    x="year",
    y="value",
    color="metric",
    markers=True,
    color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
)
fig.update_layout(
    xaxis_title="Год",
    yaxis_title="Значение",
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
    line=dict(width=3.2, shape="spline", smoothing=0.55),
    marker=dict(size=7, line=dict(color="white", width=1)),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Срез на {year_focus} год")
focus_row = country_df[country_df["year"] == year_focus]
if focus_row.empty:
    render_note("Для выбранного года в данных по этой стране нет записи.")
else:
    values = focus_row[["year"] + METRICS].rename(
        columns={
            "year": "Год",
            "gdp_pc_usd": metric_label("gdp_pc_usd"),
            "inflation_cpi": metric_label("inflation_cpi"),
            "unemployment": metric_label("unemployment"),
            "efi_total": metric_label("efi_total"),
        }
    ).fillna("—")
    st.dataframe(values, use_container_width=True)

with st.expander("Показать последние 12 лет"):
    st.dataframe(
        country_df[["year"] + METRICS]
        .tail(12)
        .rename(
            columns={
                "year": "Год",
                "gdp_pc_usd": metric_label("gdp_pc_usd"),
                "inflation_cpi": metric_label("inflation_cpi"),
                "unemployment": metric_label("unemployment"),
            }
        )
        .fillna("—"),
        use_container_width=True,
    )
