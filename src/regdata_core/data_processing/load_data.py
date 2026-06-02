import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

def load_demo_data(year: int) -> pd.DataFrame:
    """
    Temporary placeholder.
    Will be replaced with real EFI / WDI data in the next step.
    """
    # Этот модуль сейчас по сути архивный — оставлен как простой fallback с демо-структурой.
    countries = ["USA", "GBR", "DEU", "FRA", "CHN", "IND", "BRA", "RUS", "JPN", "CAN"]
    values = [70, 68, 72, 69, 60, 55, 58, 57, 74, 73]

    return pd.DataFrame(
        {
            "country": countries,
            "value": values,
            "year": year,
        }
    )
