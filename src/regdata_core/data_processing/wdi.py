from datetime import date

import pandas as pd
import wbgapi as wb

DEFAULT_INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_pc_usd",
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "SL.UEM.TOTL.ZS": "unemployment",
}

def _normalize_wb_df_wide(raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    wbgapi wide format:
    index/columns: economy + YR2020, YR2021, ...
    -> long format: iso3, year, <value_col>
    """
    df = raw.reset_index()

    if "economy" in df.columns:
        iso_col = "economy"
    elif "iso3" in df.columns:
        iso_col = "iso3"
    else:
        iso_col = df.columns[0]

    year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("YR")]
    if not year_cols:
        raise KeyError(
            f"Year columns of the form YR#### were not found. Columns: {list(df.columns)}"
        )

    out = df[[iso_col] + year_cols].melt(
        id_vars=[iso_col],
        value_vars=year_cols,
        var_name="year",
        value_name=value_col,
    )

    out = out.rename(columns={iso_col: "iso3"})
    out["year"] = out["year"].str.replace("YR", "", regex=False)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def fetch_wdi(
    indicators: dict[str, str] = DEFAULT_INDICATORS,
    start_year: int = 2000,
    end_year: int | None = None,
) -> pd.DataFrame:
    # Сначала тянем каждый показатель отдельно, потом аккуратно собираем их в общую длинную таблицу.
    if end_year is None:
        end_year = date.today().year

    years = list(range(start_year, end_year + 1))

    frames: list[pd.DataFrame] = []
    for ind_code, col_name in indicators.items():
        raw = wb.data.DataFrame(ind_code, time=years, labels=False)
        df = _normalize_wb_df_wide(raw, col_name)
        frames.append(df)

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["iso3", "year"], how="outer")

    iso3_values = sorted(out["iso3"].dropna().astype(str).unique())
    full_index = pd.MultiIndex.from_product(
        [iso3_values, years],
        names=["iso3", "year"],
    )

    out = (
        out.assign(year=out["year"].astype("Int64"))
        .set_index(["iso3", "year"])
        .reindex(full_index)
        .reset_index()
    )

    out["year"] = out["year"].astype("Int64")
    return out.sort_values(["iso3", "year"]).reset_index(drop=True)

def list_countries() -> pd.DataFrame:
    """
    Country reference table: iso3 + name.
    Supports multiple wbgapi output formats (dict generator / object generator).
    """
    rows = []
    for c in wb.economy.list():
        if isinstance(c, dict):
            iso3 = c.get("id") or c.get("iso3Code") or c.get("code")
            name = c.get("value") or c.get("name") or ""
        else:
            iso3 = getattr(c, "id", None) or getattr(c, "iso3Code", None)
            name = getattr(c, "value", "") or getattr(c, "name", "")

        if iso3:
            rows.append({"iso3": iso3, "country": name})

    return (
        pd.DataFrame(rows)
        .drop_duplicates("iso3")
        .sort_values("country")
        .reset_index(drop=True)
    )
