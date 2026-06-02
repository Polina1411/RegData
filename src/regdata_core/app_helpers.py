from __future__ import annotations

from datetime import date

import pandas as pd

from regdata_core.data_processing.cache import EFI_PATH
from regdata_core.data_processing.efi import load_efi_parquet
from regdata_core.data_processing.oecd import merge_oecd_recent_into_wdi


BASE_METRICS = ["gdp_pc_usd", "inflation_cpi", "unemployment"]
OPTIONAL_METRICS = ["efi_total"]
# Категории EFI держим в одном месте, чтобы одинаково использовать их и в таблицах, и в графиках.
EFI_BANDS = [
    (80, "Свободная", "#2f6b4f"),
    (70, "Преимущественно свободная", "#5f8f5b"),
    (60, "Умеренно свободная", "#9eb35f"),
    (50, "Преимущественно несвободная", "#d5a24f"),
    (-1, "Несвободная", "#b85c4c"),
]


def current_year() -> int:
    return date.today().year


def build_country_lookup(countries: pd.DataFrame | None) -> dict[str, str]:
    if countries is None or countries.empty:
        return {}

    cols = set(countries.columns)
    if not {"iso3", "country"}.issubset(cols):
        return {}

    clean = countries[["iso3", "country"]].dropna().drop_duplicates("iso3")
    return dict(zip(clean["iso3"].astype(str), clean["country"].astype(str)))


def country_label(iso3: str, lookup: dict[str, str]) -> str:
    name = lookup.get(str(iso3), str(iso3))
    return f"{name} ({iso3})"


def country_options(wdi: pd.DataFrame, lookup: dict[str, str]) -> list[str]:
    iso3_values = sorted(wdi["iso3"].dropna().astype(str).unique())
    return sorted(iso3_values, key=lambda iso: country_label(iso, lookup))


def latest_year_with_value(df: pd.DataFrame, metric: str) -> int | None:
    years = df.loc[df[metric].notna(), "year"].dropna()
    if years.empty:
        return None
    return int(years.max())


def latest_value_for_country(
    df: pd.DataFrame,
    iso3: str,
    metric: str,
    year: int | None = None,
) -> float | None:
    # Если передан год, берём последнее доступное значение не позже этой точки.
    subset = df[df["iso3"] == iso3]
    if year is not None:
        subset = subset[subset["year"] <= year]

    subset = subset.dropna(subset=[metric]).sort_values("year")
    if subset.empty:
        return None
    return subset.iloc[-1][metric]


def build_analytics_dataset(wdi: pd.DataFrame, oecd_recent: pd.DataFrame | None = None) -> pd.DataFrame:
    # Это центральная “склейка” данных: WDI как база, OECD как свежие наблюдения и EFI как отдельный слой.
    merged = wdi
    if oecd_recent is not None and not oecd_recent.empty:
        merged = merge_oecd_recent_into_wdi(merged, oecd_recent)

    if EFI_PATH.exists():
        efi = load_efi_parquet(EFI_PATH)
        merged = merged.merge(efi, on=["iso3", "year"], how="outer")

    return merged


def metrics_for_df(df: pd.DataFrame) -> list[str]:
    # Метрики собираем динамически, чтобы страницы не показывали пустые селекторы.
    metrics = [metric for metric in BASE_METRICS if metric in df.columns]
    for metric in OPTIONAL_METRICS:
        if metric in df.columns and df[metric].notna().any():
            metrics.append(metric)
    return metrics


def efi_category(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Нет данных"
    for threshold, label, _color in EFI_BANDS:
        if float(score) >= threshold:
            return label
    return "Нет данных"


def efi_category_color(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "#9aa5a1"
    for threshold, _label, color in EFI_BANDS:
        if float(score) >= threshold:
            return color
    return "#9aa5a1"


def format_metric_value(metric: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if metric == "gdp_pc_usd":
        return f"{float(value):,.0f}".replace(",", " ")
    return f"{float(value):.2f}"


def prepare_hover_analytics(df_year: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df_year.empty:
        return pd.DataFrame(columns=["iso3", "value", "rank", "value_text", "hover_category"])

    ranked = df_year.copy()
    ranked = ranked.sort_values("value", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["value_text"] = ranked["value"].map(lambda value: format_metric_value(metric, value))
    ranked["hover_category"] = "—"

    if metric == "efi_total":
        ranked["hover_category"] = ranked["value"].map(efi_category)

    return ranked
