from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WDI_PATH = PROCESSED_DIR / "wdi_mvp.parquet"
COUNTRIES_PATH = PROCESSED_DIR / "countries.parquet"

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
EFI_PATH = PROCESSED_DIR / "efi.parquet"
