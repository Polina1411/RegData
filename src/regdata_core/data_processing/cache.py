from pathlib import Path

import pandas as pd


# Все основные пути к локальным данным держим в одном месте, чтобы страницы не дублировали filesystem-логику.
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WDI_PATH = PROCESSED_DIR / "wdi_mvp.parquet"
COUNTRIES_PATH = PROCESSED_DIR / "countries.parquet"
RAW_COUNTRIES_GEOJSON_PATH = RAW_DIR / "countries.geojson"
LIGHT_COUNTRIES_GEOJSON_PATH = PROCESSED_DIR / "countries_light.geojson"
OECD_RECENT_PATH = PROCESSED_DIR / "oecd_recent.parquet"
EFI_PATH = PROCESSED_DIR / "efi.parquet"


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def file_version(path: Path) -> int:
    # Streamlit-кэш должен сбрасываться после обновления parquet-файлов.
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns
