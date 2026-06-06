from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    efi_category,
    efi_category_color,
    metrics_for_df,
)
from regdata_core.data_processing.cache import WDI_PATH, COUNTRIES_PATH, OECD_RECENT_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Обзор", layout="wide")
apply_app_style()
CURRENT_YEAR = date.today().year

render_hero(
    "Обзор данных",
    "Краткий обзор мировых данных: сколько стран доступно, кто находится вверху распределения и как меняется средний уровень показателя.",
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


if not WDI_PATH.exists():
    st.warning("Данные WDI не найдены. Сначала открой страницу карты и обнови данные.")
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH), file_version(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

# В обзоре мы смотрим один показатель за один год, а ниже — его среднюю мировую динамику.
available_years = wdi["year"].dropna().astype(int)
max_year = int(available_years.max()) if not available_years.empty else CURRENT_YEAR
slider_max_year = max(CURRENT_YEAR, max_year)
default_year = min(2024, slider_max_year)

col1, col2 = st.columns([1, 2])
with col1:
    metric = st.selectbox(
        "Показатель",
        metrics_for_df(wdi),
        format_func=metric_label,
    )
with col2:
    year = st.slider("Год для обзора", 2000, slider_max_year, default_year)

if slider_max_year > max_year:
    render_note(
        f"Годы {max_year + 1}-{slider_max_year} присутствуют в наборе, "
        "но часть значений за эти годы может быть ещё не опубликована."
    )

if metric in {"inflation_cpi", "unemployment"} and OECD_RECENT_PATH.exists():
    render_note(
        "Для 2025–2026 используются реальные наблюдения OECD, сведённые к последнему доступному месяцу внутри года."
    )

df_year = wdi.loc[wdi["year"] == year, ["iso3", metric]].dropna().copy()
df_year["Страна"] = df_year["iso3"].map(lambda iso: country_label(iso, country_lookup))
df_year = df_year.sort_values(metric, ascending=False)

# Это уже не рейтинг, а общий мировой фон по выбранному показателю.
metric_series = wdi[["year", metric]].dropna()
mean_by_year = metric_series.groupby("year", as_index=False)[metric].mean()

top_country = df_year.iloc[0] if not df_year.empty else None
bottom_country = df_year.iloc[-1] if not df_year.empty else None
mean_value = float(df_year[metric].mean()) if not df_year.empty else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Год обзора", year)
with c2:
    st.metric("Стран с данными", int(len(df_year)))
with c3:
    st.metric("Среднее значение", "—" if mean_value is None else f"{mean_value:.2f}")
with c4:
    st.metric("Лидер", "—" if top_country is None else top_country["Страна"])

left, right = st.columns([1.2, 1])
with left:
    st.subheader(f"Топ-10 стран: {metric_label(metric)}")
    if df_year.empty:
        st.info("Для выбранного года пока нет опубликованных значений.")
    else:
        fig_top = px.bar(
            df_year.head(10),
            x=metric,
            y="Страна",
            orientation="h",
            color_discrete_sequence=["#52796f"],
        )
        fig_top.update_layout(
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="rgba(255,255,255,0.7)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_top, use_container_width=True)

with right:
    st.subheader("Короткий вывод")
    if df_year.empty:
        render_note("На выбранный год нет достаточного числа наблюдений для краткого вывода.")
    else:
        leader_text = f"Лидер: {top_country['Страна']}" if top_country is not None else "Лидер не определён"
        outsider_text = (
            f"Нижняя граница: {bottom_country['Страна']}"
            if bottom_country is not None else
            "Нижняя граница не определена"
        )
        render_note(
            f"Показатель «{metric_label(metric)}» за {year} доступен по {len(df_year)} странам. "
            f"{leader_text}. {outsider_text}."
        )

st.subheader("Средняя мировая динамика")
fig_mean = px.area(
    mean_by_year,
    x="year",
    y=metric,
    line_group=None,
    color_discrete_sequence=["#354f52"],
)
fig_mean.update_layout(
    xaxis_title="Год",
    yaxis_title=metric_label(metric),
    plot_bgcolor="rgba(255,255,255,0.7)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
)
fig_mean.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig_mean.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig_mean.update_traces(
    line=dict(width=3, color="#2f5d50", shape="spline", smoothing=0.55),
    fillcolor="rgba(132,169,140,0.28)",
)
st.plotly_chart(fig_mean, use_container_width=True)

if metric == "efi_total" and not df_year.empty:
    # Для EFI дополнительно полезно показать не только лидеров, но и распределение стран по категориям.
    st.subheader("Категории экономической свободы")
    efi_color_map = {
        "Свободная": efi_category_color(85),
        "Преимущественно свободная": efi_category_color(75),
        "Умеренно свободная": efi_category_color(65),
        "Преимущественно несвободная": efi_category_color(55),
        "Несвободная": efi_category_color(40),
    }
    efi_groups = (
        df_year.assign(Категория=df_year["efi_total"].map(efi_category))
        .groupby("Категория", as_index=False)
        .size()
        .rename(columns={"size": "Стран"})
    )
    fig_groups = px.bar(
        efi_groups,
        x="Категория",
        y="Стран",
        color="Категория",
        color_discrete_map=efi_color_map,
    )
    fig_groups.update_layout(
        plot_bgcolor="rgba(255,255,255,0.7)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_groups, use_container_width=True)

with st.expander("Показать таблицу по странам"):
    st.dataframe(
        df_year[["Страна", metric]].rename(columns={metric: metric_label(metric)}).reset_index(drop=True),
        use_container_width=True,
    )
