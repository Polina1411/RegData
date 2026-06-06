from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    efi_category,
    efi_category_color,
)
from regdata_core.data_processing.cache import WDI_PATH, COUNTRIES_PATH, OECD_RECENT_PATH, EFI_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note


st.set_page_config(page_title="RegData — Экономическая свобода", layout="wide")
apply_app_style()

render_hero(
    "Экономическая свобода",
    "Отдельный экран для анализа индекса экономической свободы: распределение стран, лидеры, нижняя часть рейтинга и динамика по годам.",
)


def build_efi_dynamics(df: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for iso3, country_df in df[df["year"].between(year_from, year_to)].sort_values(["iso3", "year"]).groupby("iso3"):
        series = country_df[["year", "efi_total"]].dropna()
        if len(series) < 3:
            continue

        # Здесь собираем компактные признаки динамики EFI, чтобы дальше сравнивать страны уже по поведению ряда.
        values = series["efi_total"].to_numpy(dtype=float)
        years = series["year"].to_numpy(dtype=float)
        diffs = np.diff(values)
        centered_years = years - years.mean()
        denom = float((centered_years ** 2).sum()) or 1.0
        slope = float(((values - values.mean()) @ centered_years) / denom)

        rows.append(
            {
                "iso3": iso3,
                "start_value": float(values[0]),
                "end_value": float(values[-1]),
                "change": float(values[-1] - values[0]),
                "slope": slope,
                "abs_change_mean": float(np.abs(diffs).mean()),
                "diff_std": float(diffs.std()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    lower = out["abs_change_mean"].quantile(0.05)
    upper = out["abs_change_mean"].quantile(0.95)
    clipped = out["abs_change_mean"].clip(lower=lower, upper=upper)
    span = upper - lower
    # activity_score — это простая шкала от стабильных стран к более подвижным.
    out["activity_score"] = 0.0 if span == 0 else ((clipped - lower) / span).clip(0, 1)
    return out


@st.cache_data
def load_wdi_cached(path_str: str, _version: int) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


if not WDI_PATH.exists() or not EFI_PATH.exists():
    st.warning("Сначала обнови базовые данные и индекс экономической свободы на странице карты.")
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH), file_version(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

efi = wdi[["iso3", "year", "efi_total"]].dropna().copy()
if efi.empty:
    st.warning("Данные EFI не найдены в объединённом наборе.")
    st.stop()

years = sorted(efi["year"].dropna().astype(int).unique())
year = st.slider("Год", int(min(years)), int(max(years)), int(max(years)))
dyn_default_from = max(int(min(years)), int(max(years)) - 10)
dyn_from, dyn_to = st.slider("Период для динамики", int(min(years)), int(max(years)), (dyn_default_from, int(max(years))))

df_year = efi[efi["year"] == year].copy()
df_year["Страна"] = df_year["iso3"].map(lambda iso: country_label(iso, country_lookup))
df_year["Категория"] = df_year["efi_total"].map(efi_category)
dynamics_df = build_efi_dynamics(efi, dyn_from, dyn_to)
if not dynamics_df.empty:
    dynamics_df["Страна"] = dynamics_df["iso3"].map(lambda iso: country_label(iso, country_lookup))

render_note("Источник данных: The Heritage Foundation Index of Economic Freedom.")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Стран с данными", int(len(df_year)))
with c2:
    st.metric("Средний индекс", f"{df_year['efi_total'].mean():.2f}" if not df_year.empty else "—")
with c3:
    best = df_year.sort_values("efi_total", ascending=False).head(1)
    st.metric("Лидер", best.iloc[0]["Страна"] if not best.empty else "—")

if not df_year.empty:
    top_country_name = df_year.sort_values("efi_total", ascending=False).iloc[0]["Страна"]
    bottom_country_name = df_year.sort_values("efi_total", ascending=True).iloc[0]["Страна"]
    render_note(
        f"Краткий вывод для защиты: в {year} году лидер по индексу экономической свободы — {top_country_name}, "
        f"а нижняя часть распределения замыкается страной {bottom_country_name}. "
        "Это даёт наглядный срез различий в качестве институтов и экономической среды между странами."
    )

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Топ-15 стран")
    top15 = df_year.sort_values("efi_total", ascending=False).head(15)
    fig_top = px.bar(
        top15,
        x="efi_total",
        y="Страна",
        orientation="h",
        color="efi_total",
        color_continuous_scale=["#d8e2dc", "#a3b18a", "#588157", "#344e41"],
    )
    fig_top.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_top, use_container_width=True)

with right:
    st.subheader("Распределение по категориям")
    color_map = {
        "Свободная": efi_category_color(85),
        "Преимущественно свободная": efi_category_color(75),
        "Умеренно свободная": efi_category_color(65),
        "Преимущественно несвободная": efi_category_color(55),
        "Несвободная": efi_category_color(40),
    }
    groups = df_year.groupby("Категория", as_index=False).size().rename(columns={"size": "Стран"})
    fig_groups = px.pie(
        groups,
        names="Категория",
        values="Стран",
        color="Категория",
        color_discrete_map=color_map,
        hole=0.46,
    )
    fig_groups.update_layout(
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="",
    )
    st.plotly_chart(fig_groups, use_container_width=True)

st.subheader("Глобальная динамика индекса")
world_mean = efi.groupby("year", as_index=False)["efi_total"].mean()
fig_mean = px.line(
    world_mean,
    x="year",
    y="efi_total",
    markers=True,
    color_discrete_sequence=["#2f5d50"],
)
fig_mean.update_layout(
    xaxis_title="Год",
    yaxis_title="Индекс экономической свободы",
    plot_bgcolor="rgba(255,255,255,0.74)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
fig_mean.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig_mean.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig_mean.update_traces(
    line=dict(width=3.2, shape="spline", smoothing=0.55),
    marker=dict(size=7, line=dict(color="white", width=1)),
)
st.plotly_chart(fig_mean, use_container_width=True)

st.subheader("Динамика стран")
render_note(
    "Здесь индекс рассматривается как временной ряд: кто почти не меняется, у кого плавный рост, а у кого динамика заметно более резкая."
)

if dynamics_df.empty:
    st.info("Для выбранного периода недостаточно данных, чтобы оценить динамику стран.")
else:
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Стран в динамике", int(len(dynamics_df)))
    with d2:
        st.metric("Средняя подвижность", f"{dynamics_df['activity_score'].mean():.2f}")
    with d3:
        mover = dynamics_df.sort_values("activity_score", ascending=False).iloc[0]
        st.metric("Самая подвижная", mover["Страна"])
    with d4:
        stable = dynamics_df.sort_values("activity_score", ascending=True).iloc[0]
        st.metric("Самая стабильная", stable["Страна"])

    fig_dyn = px.scatter(
        dynamics_df,
        x="start_value",
        y="end_value",
        color="activity_score",
        hover_name="Страна",
        size="abs_change_mean",
        size_max=24,
        color_continuous_scale=["#d8ece4", "#7fb3a3", "#1f6b57"],
        hover_data={"change": ":.2f", "slope": ":.3f", "activity_score": ":.2f"},
    )
    fig_dyn.update_layout(
        xaxis_title=f"Индекс в начале периода ({dyn_from})",
        yaxis_title=f"Индекс в конце периода ({dyn_to})",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar_title="Подвижность",
    )
    fig_dyn.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig_dyn.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig_dyn.update_traces(marker=dict(opacity=0.88, line=dict(color="white", width=1.2)))
    st.plotly_chart(fig_dyn, use_container_width=True)

    left_dyn, right_dyn = st.columns([1, 1])
    with left_dyn:
        st.subheader("Самые сильные изменения")
        st.dataframe(
            dynamics_df[["Страна", "change", "activity_score"]]
            .rename(columns={"change": "Изменение индекса", "activity_score": "Подвижность"})
            .sort_values("Подвижность", ascending=False)
            .head(12)
            .round(3),
            use_container_width=True,
            hide_index=True,
        )
    with right_dyn:
        st.subheader("Самые стабильные страны")
        st.dataframe(
            dynamics_df[["Страна", "change", "activity_score"]]
            .rename(columns={"change": "Изменение индекса", "activity_score": "Подвижность"})
            .sort_values("Подвижность", ascending=True)
            .head(12)
            .round(3),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Показать таблицу"):
    export_df = (
        df_year[["Страна", "efi_total", "Категория"]]
        .rename(columns={"efi_total": "Индекс экономической свободы"})
        .sort_values("Индекс экономической свободы", ascending=False)
        .reset_index(drop=True)
    )
    st.download_button(
        "Скачать таблицу CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"regdata_efi_{year}.csv",
        mime="text/csv",
    )
    st.dataframe(export_df, use_container_width=True)
