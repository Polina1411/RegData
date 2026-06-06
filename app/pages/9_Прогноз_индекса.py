from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.analytics.efi_forecast import forecast_country_components, weighted_index_from_components
from regdata_core.app_helpers import build_country_lookup, country_label
from regdata_core.data_processing.cache import COUNTRIES_PATH, EFI_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Прогноз индекса", layout="wide")
apply_app_style()

render_hero(
    "Прогноз индекса экономической свободы",
    "Простой прогноз на основе последних наблюдений: для каждой компоненты берётся среднее годовое изменение, "
    "после чего пересчитывается индекс по выбранным весам.",
)

EFI_COMPONENTS = [
    "property_rights",
    "government_integrity",
    "judicial_effectiveness",
    "tax_burden",
    "government_spending",
    "fiscal_health",
    "business_freedom",
    "labor_freedom",
    "monetary_freedom",
    "trade_freedom",
    "investment_freedom",
    "financial_freedom",
]


@st.cache_data
def load_efi_cached(path_str: str, _version: int) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


def build_weights() -> dict[str, float]:
    weight_cols = st.columns(3)
    weights_raw: dict[str, int] = {}

    for idx, component in enumerate(EFI_COMPONENTS):
        with weight_cols[idx % 3]:
            weights_raw[component] = st.slider(
                metric_label(component),
                0,
                100,
                10,
                1,
                key=f"forecast_{component}",
            )

    total_weight = sum(weights_raw.values())
    if total_weight == 0:
        st.warning("Нужно задать хотя бы один ненулевой вес.")
        st.stop()

    return {key: value / total_weight for key, value in weights_raw.items()}


def build_plot_frame(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [
            actual_df[["year", "official_index", "baseline_index", "custom_index", "Тип"]],
            forecast_df[["year", "official_index", "baseline_index", "custom_index", "Тип"]],
        ],
        ignore_index=True,
    )

    plot_df = combined.melt(
        id_vars=["year", "Тип"],
        value_vars=["official_index", "baseline_index", "custom_index"],
        var_name="series",
        value_name="value",
    )
    plot_df["Ряд"] = plot_df["series"].map(
        {
            "official_index": "Официальный индекс",
            "baseline_index": "Прогноз с равными весами",
            "custom_index": "Прогноз с твоими весами",
        }
    )
    return plot_df.dropna(subset=["value"])


if not EFI_PATH.exists():
    st.warning("Сначала обнови индекс экономической свободы на странице карты.")
    st.stop()

efi = load_efi_cached(str(EFI_PATH), file_version(EFI_PATH))
countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

forecast_ready = (
    efi.dropna(subset=EFI_COMPONENTS)
    .groupby("iso3", as_index=False)
    .size()
    .rename(columns={"size": "rows_with_components"})
)
forecast_ready = forecast_ready[forecast_ready["rows_with_components"] >= 2]
iso_options = sorted(
    forecast_ready["iso3"].dropna().astype(str).unique(),
    key=lambda iso: country_label(iso, country_lookup),
)

if not iso_options:
    st.warning("Нет стран с достаточной историей для прогноза.")
    st.stop()

col1, col2 = st.columns([1.2, 1])
with col1:
    country_iso = st.selectbox(
        "Страна",
        iso_options,
        format_func=lambda iso: country_label(iso, country_lookup),
    )
with col2:
    horizon = st.slider("Горизонт прогноза, лет", 1, 5, 3)

st.subheader("Веса компонент")
weights = build_weights()

country_df = efi[efi["iso3"] == country_iso].sort_values("year").copy()
history_df = country_df.dropna(subset=EFI_COMPONENTS)
if len(history_df) < 2:
    st.warning("Для выбранной страны недостаточно истории для прогноза.")
    st.stop()

forecast_df = forecast_country_components(history_df, EFI_COMPONENTS, horizon=horizon, lookback_years=5)
forecast_df["custom_index"] = weighted_index_from_components(forecast_df, weights, EFI_COMPONENTS)
forecast_df["baseline_index"] = weighted_index_from_components(
    forecast_df,
    {component: 1 / len(EFI_COMPONENTS) for component in EFI_COMPONENTS},
    EFI_COMPONENTS,
)
forecast_df["official_index"] = pd.NA
forecast_df["Тип"] = "Прогноз"

actual_df = history_df[["year", "efi_total"] + EFI_COMPONENTS].copy()
actual_df["custom_index"] = weighted_index_from_components(actual_df, weights, EFI_COMPONENTS)
actual_df["baseline_index"] = weighted_index_from_components(
    actual_df,
    {component: 1 / len(EFI_COMPONENTS) for component in EFI_COMPONENTS},
    EFI_COMPONENTS,
)
actual_df["official_index"] = actual_df["efi_total"]
actual_df["Тип"] = "Факт"

plot_df = build_plot_frame(actual_df, forecast_df)
country_name = country_label(country_iso, country_lookup)

last_actual = actual_df.iloc[-1]
first_pred = forecast_df.iloc[0]
last_pred = forecast_df.iloc[-1]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Последний фактический индекс", f"{last_actual['efi_total']:.2f}", f"{int(last_actual['year'])}")
with c2:
    st.metric("Первый прогнозный год", f"{first_pred['custom_index']:.2f}", f"{int(first_pred['year'])}")
with c3:
    st.metric("Конец горизонта", f"{last_pred['custom_index']:.2f}", f"{int(last_pred['year'])}")

st.subheader(country_name)
fig = px.line(
    plot_df,
    x="year",
    y="value",
    color="Ряд",
    line_dash="Тип",
    markers=True,
    color_discrete_sequence=["#2f3e46", "#84a98c", "#1f6b57"],
)
fig.update_layout(
    xaxis_title="Год",
    yaxis_title="Индекс",
    plot_bgcolor="rgba(255,255,255,0.74)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend_title_text="",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
st.plotly_chart(fig, use_container_width=True)

forecast_table = forecast_df[["year", "baseline_index", "custom_index"]].rename(
    columns={
        "year": "Год",
        "baseline_index": "Индекс с равными весами",
        "custom_index": "Индекс с твоими весами",
    }
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Прогноз по годам")
    st.dataframe(forecast_table.round(2), use_container_width=True, hide_index=True)
with right:
    st.subheader("Последние фактические значения")
    actual_tail = actual_df[["year", "official_index", "custom_index"]].tail(5).rename(
        columns={
            "year": "Год",
            "official_index": "Официальный индекс",
            "custom_index": "Индекс с твоими весами",
        }
    )
    st.dataframe(actual_tail.round(2), use_container_width=True, hide_index=True)

with st.expander("Показать прогноз по компонентам"):
    component_table = forecast_df[["year"] + EFI_COMPONENTS].rename(
        columns={"year": "Год", **{component: metric_label(component) for component in EFI_COMPONENTS}}
    )
    st.dataframe(component_table.round(2), use_container_width=True, hide_index=True)
