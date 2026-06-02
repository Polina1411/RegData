from __future__ import annotations

import io
import ssl
import urllib.request
from datetime import date

import pandas as pd


OECD_HEADERS = {"User-Agent": "Mozilla/5.0"}

OECD_UNEMPLOYMENT_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC/"
    ".UNE_LF_M...Y._T.Y_GE15..M"
    "?startPeriod=2025-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)

OECD_INFLATION_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL/"
    ".M.N.CPI.PA._T.N.GY"
    "?startPeriod=2025-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
)


def _read_oecd_csv(url: str) -> pd.DataFrame:
    request = urllib.request.Request(url, headers=OECD_HEADERS)
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=90) as response:
        text = response.read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def _latest_month_per_year(df: pd.DataFrame, value_name: str, period_name: str) -> pd.DataFrame:
    # Для годового слоя берём последнее доступное месячное наблюдение внутри каждого года.
    out = df.copy()
    out["period_dt"] = pd.to_datetime(out["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    out["year"] = out["period_dt"].dt.year.astype("Int64")
    out = out.dropna(subset=["year", "period_dt", "OBS_VALUE"])
    out = out.sort_values(["REF_AREA", "year", "period_dt"])
    out = out.groupby(["REF_AREA", "year"], as_index=False).tail(1)
    out = out.rename(columns={"REF_AREA": "iso3", "OBS_VALUE": value_name, "TIME_PERIOD": period_name})
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out[["iso3", "year", value_name, period_name]].reset_index(drop=True)


def fetch_oecd_recent() -> pd.DataFrame:
    inflation_raw = _read_oecd_csv(OECD_INFLATION_URL)
    unemployment_raw = _read_oecd_csv(OECD_UNEMPLOYMENT_URL)

    inflation = _latest_month_per_year(
        inflation_raw,
        value_name="inflation_cpi",
        period_name="inflation_period",
    )
    unemployment = _latest_month_per_year(
        unemployment_raw,
        value_name="unemployment",
        period_name="unemployment_period",
    )

    out = inflation.merge(unemployment, on=["iso3", "year"], how="outer")
    out = out[out["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    out["year"] = out["year"].astype("Int64")
    out["source"] = "OECD"
    out["fetched_year"] = date.today().year
    return out.sort_values(["iso3", "year"]).reset_index(drop=True)


def merge_oecd_recent_into_wdi(
    wdi: pd.DataFrame,
    oecd_recent: pd.DataFrame,
    start_year: int = 2025,
) -> pd.DataFrame:
    # OECD используем как свежую надстройку над WDI только для последних лет и только по нужным метрикам.
    merged = wdi.merge(
        oecd_recent,
        on=["iso3", "year"],
        how="outer",
        suffixes=("", "_oecd"),
    )

    recent_mask = merged["year"].fillna(0).astype(int) >= start_year
    for metric in ["inflation_cpi", "unemployment"]:
        oecd_col = f"{metric}_oecd"
        if oecd_col in merged.columns:
            merged.loc[recent_mask & merged[oecd_col].notna(), metric] = merged.loc[
                recent_mask & merged[oecd_col].notna(),
                oecd_col,
            ]

    merged["year"] = merged["year"].astype("Int64")
    return merged
