from datetime import date

import streamlit as st
import pandas as pd
import plotly.express as px

from regdata_core.app_helpers import build_analytics_dataset, build_country_lookup, country_label, country_options, metrics_for_df
from regdata_core.data_processing.cache import WDI_PATH, OECD_RECENT_PATH, COUNTRIES_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label

st.set_page_config(page_title="RegData — Корреляция", layout="wide")
apply_app_style()
CURRENT_YEAR = date.today().year
render_hero(
    "Корреляционный анализ",
    "Посмотри, как связаны два показателя внутри одной страны за выбранный период.",
)

if not WDI_PATH.exists():
    st.warning("Данные WDI не найдены. Сначала открой страницу карты и обнови данные.")
    st.stop()

@st.cache_data
def load_wdi(_version: int):
    return load_parquet(WDI_PATH)

@st.cache_data
def load_countries(_version: int):
    if COUNTRIES_PATH.exists():
        return load_parquet(COUNTRIES_PATH)
    return pd.DataFrame(columns=["iso3", "country"])

wdi = load_wdi(file_version(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
countries = load_countries(file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

st.subheader("Параметры")

metrics = metrics_for_df(wdi)

iso_candidates = country_options(wdi, country_lookup)
default_iso3 = st.session_state.get("selected_iso3")
if default_iso3 not in iso_candidates:
    default_iso3 = iso_candidates[0]

col1, col2, col3 = st.columns(3)

with col1:
    iso3 = st.selectbox(
        "Страна",
        iso_candidates,
        index=iso_candidates.index(default_iso3),
        format_func=lambda iso: country_label(iso, country_lookup),
    )

with col2:
    x_metric = st.selectbox("Показатель по оси X", metrics, index=0, format_func=metric_label)

with col3:
    y_metric = st.selectbox("Показатель по оси Y", metrics, index=2, format_func=metric_label)

if x_metric == y_metric:
    st.warning("Выбери разные показатели для осей X и Y, иначе корреляция будет всегда равна 1.")
    st.stop()

min_year = int(wdi["year"].dropna().min())
max_year = int(wdi["year"].dropna().max())
slider_max_year = max(CURRENT_YEAR, max_year)
year_from, year_to = st.slider("Период", min_year, slider_max_year, (min_year, slider_max_year))

if slider_max_year > max_year:
    render_note(
        f"Годы {max_year + 1}-{slider_max_year} доступны в диапазоне, "
        "но значения World Bank за эти годы могут ещё отсутствовать."
    )

if OECD_RECENT_PATH.exists() and ({x_metric, y_metric} & {"inflation_cpi", "unemployment"}):
    render_note(
        "Для инфляции и безработицы за 2025–2026 используются реальные месячные наблюдения OECD."
    )

df_all_years = wdi[
    (wdi["iso3"] == iso3) &
    (wdi["year"].between(year_from, year_to))
][["year", x_metric, y_metric]].sort_values("year")

df = df_all_years.dropna().sort_values("year")

if df.empty:
    st.warning("Нет данных для выбранной страны и периода.")
    st.stop()

# Корреляция здесь считается только по тем годам, где есть оба показателя сразу.
st.subheader(f"{metric_label(x_metric)} и {metric_label(y_metric)}")
st.caption(country_label(iso3, country_lookup))
render_note(
    "Каждая точка — это один год. Чем правее и выше точка, тем больше значения обоих показателей в этот год."
)

plot_df = df[[x_metric, y_metric]].copy()
plot_df["year"] = df["year"].astype(int)
fig = px.scatter(
    plot_df,
    x=x_metric,
    y=y_metric,
    color="year",
    color_continuous_scale=["#d8ece4", "#7fb3a3", "#1f6b57"],
    hover_data={"year": True, x_metric: ":.2f", y_metric: ":.2f"},
)
fig.update_layout(
    xaxis_title=metric_label(x_metric),
    yaxis_title=metric_label(y_metric),
    plot_bgcolor="rgba(255,255,255,0.7)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="closest",
    coloraxis_colorbar_title="Год",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_traces(
    marker=dict(size=13, opacity=0.9, line=dict(color="rgba(255,255,255,0.9)", width=1.5)),
    hovertemplate=(
        "Год: %{customdata[0]}<br>"
        + f"{metric_label(x_metric)}: %{{x:.2f}}<br>"
        + f"{metric_label(y_metric)}: %{{y:.2f}}"
        + "<extra></extra>"
    ),
    customdata=plot_df[["year"]].to_numpy(),
)
st.plotly_chart(fig, use_container_width=True)

corr = df[[x_metric, y_metric]].corr().iloc[0, 1]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Наблюдений", int(len(df)))
with c2:
    st.metric("Корреляция Пирсона", f"{corr:.3f}")
with c3:
    st.metric("Период", f"{year_from}–{year_to}")

with st.expander("Показать таблицу данных"):
    st.dataframe(
        df_all_years.rename(
            columns={
                "year": "Год",
                x_metric: metric_label(x_metric),
                y_metric: metric_label(y_metric),
            }
        ).fillna("—"),
        use_container_width=True,
    )
