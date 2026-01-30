from __future__ import annotations
from pathlib import Path
import pandas as pd

# Мы будем хранить сырой файл в data/raw и собранный parquet в data/processed
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EFI_RAW_PATH = RAW_DIR / "efi.csv"
EFI_PATH = PROCESSED_DIR / "efi.parquet"

def load_efi_csv(path: Path = EFI_RAW_PATH) -> pd.DataFrame:
    """
    Ожидаем таблицу с минимум:
    - iso3 (или ISO3)
    - year (или Year)
    - efi_total (или Score / Overall Score)
    Возвращаем формат: iso3, year, efi_total
    """
    df = pd.read_csv(path)

    # приведение названий колонок к нижнему регистру без пробелов
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # возможные варианты колонок
    iso_candidates = ["iso3", "ISO3", "country_code", "Country Code"]
    year_candidates = ["year", "Year"]
    score_candidates = ["efi_total", "score", "Score", "overall_score", "Overall Score", "overall", "Overall"]

    iso_col = next((c for c in iso_candidates if c in df.columns), None)
    year_col = next((c for c in year_candidates if c in df.columns), None)
    score_col = next((c for c in score_candidates if c in df.columns), None)

    if iso_col is None or year_col is None or score_col is None:
        raise ValueError(
            f"Не нашла нужные колонки в EFI CSV.\n"
            f"Нужны iso3/year/score.\n"
            f"Колонки файла: {list(df.columns)}"
        )

    out = df[[iso_col, year_col, score_col]].rename(
        columns={iso_col: "iso3", year_col: "year", score_col: "efi_total"}
    )

    out["iso3"] = out["iso3"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["efi_total"] = pd.to_numeric(out["efi_total"], errors="coerce")

    out = out.dropna(subset=["iso3", "year", "efi_total"]).reset_index(drop=True)
    return out

def save_efi_parquet(df: pd.DataFrame, path: Path = EFI_PATH) -> None:
    df.to_parquet(path, index=False)

def load_efi_parquet(path: Path = EFI_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
