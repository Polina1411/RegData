from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.app_helpers import build_country_lookup, country_label
from regdata_core.data_processing.cache import COUNTRIES_PATH, EFI_PATH, file_version, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Конструктор индекса", layout="wide")
apply_app_style()

render_hero(
    "Конструктор индекса экономической свободы",
    "Выбирай веса для 12 компонент, считай свой вариант индекса и сравнивай его с официальным индексом Heritage.",
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


def calculate_weighted_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    # Считаем индекс только по тем компонентам, где у страны есть данные.
    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    available_weight = pd.Series(0.0, index=df.index, dtype=float)

    for component in EFI_COMPONENTS:
        values = pd.to_numeric(df[component], errors="coerce")
        has_value = values.notna().astype(float)
        weighted_sum = weighted_sum + values.fillna(0.0) * weights[component]
        available_weight = available_weight + has_value * weights[component]

    return weighted_sum / available_weight.where(available_weight > 0)


def build_weight_table(weights: dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "Компонента": [metric_label(component) for component in EFI_COMPONENTS],
                "Нормированный вес": [weights[component] for component in EFI_COMPONENTS],
            }
        )
        .sort_values("Нормированный вес", ascending=False)
        .reset_index(drop=True)
    )


def build_rank_shift_table(df_year: pd.DataFrame) -> pd.DataFrame:
    out = df_year[["Страна", "efi_total", "custom_index", "official_rank", "custom_rank", "rank_diff"]].copy()
    out["Сдвиг места"] = out["rank_diff"].map(
        lambda value: "Без изменений" if value == 0 else f"{int(abs(value))} {'выше' if value < 0 else 'ниже'}"
    )
    return out.rename(
        columns={
            "efi_total": "Официальный индекс",
            "custom_index": "Кастомный индекс",
            "official_rank": "Официальное место",
            "custom_rank": "Кастомное место",
        }
    )


def finite_weight_sensitivity(row: pd.Series, weights: dict[str, float], epsilon: float = 0.01) -> pd.DataFrame:
    # Небольшой what-if: чуть увеличиваем один вес и смотрим, насколько меняется итог.
    base_index = calculate_weighted_index(pd.DataFrame([row]), weights).iloc[0]
    rows: list[dict[str, float | str]] = []

    for component in EFI_COMPONENTS:
        bumped = weights.copy()
        bumped[component] += epsilon
        total = sum(bumped.values())
        bumped = {key: value / total for key, value in bumped.items()}
        bumped_index = calculate_weighted_index(pd.DataFrame([row]), bumped).iloc[0]
        rows.append(
            {
                "Компонента": metric_label(component),
                "Значение": float(row[component]),
                "Эффект на индекс": float(bumped_index - base_index),
            }
        )

    return pd.DataFrame(rows).sort_values("Эффект на индекс", ascending=False).reset_index(drop=True)


@st.cache_data
def load_efi_cached(path_str: str, _version: int) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str, _version: int) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


if not EFI_PATH.exists():
    st.warning("Сначала обнови индекс экономической свободы на странице карты.")
    st.stop()

efi = load_efi_cached(str(EFI_PATH), file_version(EFI_PATH))
missing_components = [col for col in EFI_COMPONENTS if col not in efi.columns]
if missing_components:
    st.warning("EFI загружен в старом формате без компонент. Нажми «Обновить индекс экономической свободы» на странице карты.")
    st.stop()

countries = load_countries_cached(str(COUNTRIES_PATH), file_version(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

years = sorted(efi["year"].dropna().astype(int).unique())
default_year = int(years[-1])

col1, col2 = st.columns([1.2, 1])
with col1:
    year = st.slider("Год", int(min(years)), int(max(years)), default_year)
with col2:
    year_country_df = efi[efi["year"] == year].dropna(subset=["efi_total"] + EFI_COMPONENTS)
    iso_options = sorted(
        year_country_df["iso3"].dropna().astype(str).unique(),
        key=lambda iso: country_label(iso, country_lookup),
    )
    if not iso_options:
        st.warning("Для выбранного года нет достаточно полных данных по странам.")
        st.stop()
    country_iso = st.selectbox(
        "Страна для примера",
        iso_options,
        format_func=lambda iso: country_label(iso, country_lookup),
    )

st.subheader("Настройка весов")
render_note("Вес показывает относительную важность компоненты. После выбора все веса автоматически нормируются.")

weight_cols = st.columns(3)
weights_raw: dict[str, int] = {}
for idx, component in enumerate(EFI_COMPONENTS):
    with weight_cols[idx % 3]:
        weights_raw[component] = st.slider(metric_label(component), 0, 100, 10, 1, key=f"builder_{component}")

total_weight = sum(weights_raw.values())
if total_weight == 0:
    st.warning("Нужно задать хотя бы один ненулевой вес.")
    st.stop()

weights = {key: value / total_weight for key, value in weights_raw.items()}

df_year = efi[efi["year"] == year].copy()
df_year["custom_index"] = calculate_weighted_index(df_year, weights)
df_year = df_year.dropna(subset=["efi_total", "custom_index"])
df_year["official_rank"] = df_year["efi_total"].rank(ascending=False, method="min")
df_year["custom_rank"] = df_year["custom_index"].rank(ascending=False, method="min")
df_year["rank_diff"] = df_year["custom_rank"] - df_year["official_rank"]
df_year["Страна"] = df_year["iso3"].map(lambda iso: country_label(iso, country_lookup))

if df_year.empty:
    st.warning("Нет данных для выбранного года.")
    st.stop()

correlation = float(df_year[["efi_total", "custom_index"]].corr().iloc[0, 1])
mae = float((df_year["efi_total"] - df_year["custom_index"]).abs().mean())
mean_rank_shift = float(df_year["rank_diff"].abs().mean())
country_row = df_year[df_year["iso3"] == country_iso]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Корреляция с официальным индексом", f"{correlation:.3f}")
with c2:
    st.metric("Среднее отклонение", f"{mae:.2f}")
with c3:
    st.metric("Средний сдвиг мест", f"{mean_rank_shift:.1f}")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Кастомный индекс и официальный")
    fig = px.scatter(
        df_year,
        x="efi_total",
        y="custom_index",
        hover_name="Страна",
        hover_data={"official_rank": True, "custom_rank": True},
        color="rank_diff",
        color_continuous_scale=["#a53f2b", "#efe4d8", "#1f6b57"],
    )
    fig.update_layout(
        xaxis_title="Официальный индекс",
        yaxis_title="Кастомный индекс",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar_title="Сдвиг мест",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Сильнее всего меняются")
    rank_shift_table = build_rank_shift_table(df_year)
    st.dataframe(
        rank_shift_table.sort_values("rank_diff", key=lambda series: series.abs(), ascending=False).head(12).drop(columns=["rank_diff"]),
        use_container_width=True,
        hide_index=True,
    )

if not country_row.empty:
    row = country_row.iloc[0]

    st.subheader(country_label(country_iso, country_lookup))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Официальный индекс", f"{row['efi_total']:.2f}", f"место: {int(row['official_rank'])}")
    with c2:
        st.metric("Кастомный индекс", f"{row['custom_index']:.2f}", f"место: {int(row['custom_rank'])}")
    with c3:
        shift_value = int(abs(row["rank_diff"]))
        shift_text = "без изменений" if shift_value == 0 else f"{shift_value} поз."
        st.metric("Сдвиг места", shift_text)

    sensitivity_df = finite_weight_sensitivity(row, weights)
    fig_sensitivity = px.bar(
        sensitivity_df,
        x="Эффект на индекс",
        y="Компонента",
        orientation="h",
        color="Эффект на индекс",
        color_continuous_scale=["#d8ece4", "#7fb3a3", "#1f6b57"],
    )
    fig_sensitivity.update_layout(
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sensitivity, use_container_width=True)

st.subheader("Нормированные веса")
st.dataframe(build_weight_table(weights), use_container_width=True, hide_index=True)

with st.expander("Показать полную таблицу по странам"):
    st.dataframe(
        build_rank_shift_table(df_year).drop(columns=["rank_diff"]),
        use_container_width=True,
        hide_index=True,
    )
