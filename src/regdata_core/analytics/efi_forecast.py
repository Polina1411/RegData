from __future__ import annotations

import numpy as np
import pandas as pd


def _recent_yearly_change(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.diff(values).mean())


def forecast_country_components(
    country_df: pd.DataFrame,
    component_cols: list[str],
    horizon: int,
    lookback_years: int = 5,
) -> pd.DataFrame:
    # Простой прогноз: продолжаем недавний средний годовой темп изменения каждой компоненты.
    base = country_df.sort_values("year").dropna(subset=component_cols).copy()
    if len(base) < 2:
        raise ValueError("Not enough history for selected country.")

    recent = base.tail(lookback_years)
    last_year = int(recent["year"].max())
    last_values = recent[component_cols].iloc[-1]
    changes = {
        col: _recent_yearly_change(recent[col].to_numpy(dtype=float))
        for col in component_cols
    }

    rows: list[dict[str, float | int]] = []
    for step in range(1, horizon + 1):
        row: dict[str, float | int] = {"year": last_year + step}
        for col in component_cols:
            forecast_value = float(last_values[col]) + changes[col] * step
            row[col] = float(np.clip(forecast_value, 0.0, 100.0))
        rows.append(row)

    return pd.DataFrame(rows)


def weighted_index_from_components(
    df: pd.DataFrame,
    weights: dict[str, float],
    component_cols: list[str],
) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    available_weight = pd.Series(0.0, index=df.index, dtype=float)

    for col in component_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        has_value = values.notna().astype(float)
        weighted_sum = weighted_sum + values.fillna(0.0) * weights[col]
        available_weight = available_weight + has_value * weights[col]

    return weighted_sum / available_weight.where(available_weight > 0)
