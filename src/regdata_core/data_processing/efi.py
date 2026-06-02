from __future__ import annotations
import io
from pathlib import Path
import ssl
import urllib.request
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EFI_RAW_PATH = RAW_DIR / "efi.csv"
EFI_PATH = PROCESSED_DIR / "efi.parquet"
EFI_SCORES_URL = "https://economicfreedom.heritage.org/assets/data/csv/ef-country-scores.csv"
EFI_NAMES_URL = "https://economicfreedom.heritage.org/assets/data/csv/ef-country-names.csv"
EFI_HEADERS = {"User-Agent": "Mozilla/5.0"}
EFI_COMPONENT_MAP = {
    "Overall": "efi_total",
    "Property Rights": "property_rights",
    "Government Integrity": "government_integrity",
    "Judicial Effectiveness": "judicial_effectiveness",
    "Tax Burden": "tax_burden",
    "Government Spending": "government_spending",
    "Fiscal Health": "fiscal_health",
    "Business Freedom": "business_freedom",
    "Labor Freedom": "labor_freedom",
    "Monetary Freedom": "monetary_freedom",
    "Trade Freedom": "trade_freedom",
    "Investment Freedom": "investment_freedom",
    "Financial Freedom": "financial_freedom",
}

def normalize_efi_df(df: pd.DataFrame) -> pd.DataFrame:
    # У Heritage названия столбцов могут немного плавать, поэтому приводим всё к одному внутреннему формату.
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    df = df.rename(columns={src: dest for src, dest in EFI_COMPONENT_MAP.items() if src in df.columns})

    iso_candidates = ["iso3", "ISO3", "country_code", "Country Code"]
    year_candidates = ["year", "Year"]
    score_candidates = [
        "efi_total",
        "score",
        "Score",
        "overall_score",
        "Overall Score",
        "overall",
        "Overall",
    ]

    iso_col = next((c for c in iso_candidates if c in df.columns), None)
    year_col = next((c for c in year_candidates if c in df.columns), None)
    score_col = next((c for c in score_candidates if c in df.columns), None)

    if iso_col is None or year_col is None or score_col is None:
        raise ValueError(
            "Required columns not found in EFI CSV. "
            "Expected iso3 / year / score. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[iso_col, year_col, score_col]].rename(
        columns={
            iso_col: "iso3",
            year_col: "year",
            score_col: "efi_total",
        }
    )

    out["iso3"] = out["iso3"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["efi_total"] = pd.to_numeric(out["efi_total"], errors="coerce")

    extra_cols = [col for col in EFI_COMPONENT_MAP.values() if col in df.columns and col not in out.columns]
    if extra_cols:
        extra_df = df[[iso_col, year_col] + extra_cols].rename(
            columns={iso_col: "iso3", year_col: "year"}
        )
        for col in extra_cols:
            extra_df[col] = pd.to_numeric(extra_df[col], errors="coerce")
        out = out.merge(extra_df, on=["iso3", "year"], how="left")

    out = out.dropna(subset=["iso3", "year", "efi_total"]).reset_index(drop=True)
    return out


def _read_remote_csv(url: str) -> pd.DataFrame:
    request = urllib.request.Request(url, headers=EFI_HEADERS)
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=90) as response:
        text = response.read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def fetch_efi_official() -> pd.DataFrame:
    # Официальный EFI тянем из двух CSV: со значениями и со справочником названий/ISO-кодов.
    scores = _read_remote_csv(EFI_SCORES_URL)
    names = _read_remote_csv(EFI_NAMES_URL)

    merged = scores.merge(
        names[["name_web", "name_ISO3166_3"]],
        on="name_web",
        how="left",
    )

    merged = merged.rename(
        columns={
            "name_ISO3166_3": "iso3",
            "Year": "year",
        }
    )
    return normalize_efi_df(merged)

def load_efi_csv(path: Path = EFI_RAW_PATH) -> pd.DataFrame:
    """
    Expected columns (at minimum):
    - iso3 (or ISO3)
    - year (or Year)
    - efi_total (or Score / Overall Score)

    Output format:
    iso3, year, efi_total
    """
    df = pd.read_csv(path)
    return normalize_efi_df(df)

def save_efi_parquet(df: pd.DataFrame, path: Path = EFI_PATH) -> None:
    df.to_parquet(path, index=False)

def load_efi_parquet(path: Path = EFI_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
