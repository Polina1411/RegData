from __future__ import annotations

import numpy as np
import pandas as pd


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def build_forecast_training_data(
    df: pd.DataFrame,
    component_cols: list[str],
    window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    # Каждую страну режем на скользящие окна истории, чтобы учить модель предсказывать следующий шаг.
    samples_x: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []

    for _iso3, country_df in df.sort_values(["iso3", "year"]).groupby("iso3"):
        country_df = country_df.dropna(subset=component_cols).copy()
        if len(country_df) <= window:
            continue

        values = country_df[component_cols].to_numpy(dtype=float)
        for idx in range(window, len(values)):
            history = values[idx - window:idx].reshape(-1)
            target = values[idx]
            samples_x.append(history)
            samples_y.append(target)

    if not samples_x:
        raise ValueError("Not enough EFI history to train forecast model.")

    return np.vstack(samples_x), np.vstack(samples_y)


def train_component_forecaster(
    df: pd.DataFrame,
    component_cols: list[str],
    window: int = 3,
    hidden_size: int = 32,
    epochs: int = 1500,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> dict:
    # Здесь модель учится предсказывать не итоговый индекс, а сразу следующий набор EFI-компонент.
    x, y = build_forecast_training_data(df, component_cols, window=window)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std == 0, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std == 0, 1.0, y_std)
    y_scaled = (y - y_mean) / y_std

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.15, size=(x_scaled.shape[1], hidden_size))
    b1 = np.zeros((1, hidden_size))
    w2 = rng.normal(0, 0.15, size=(hidden_size, y_scaled.shape[1]))
    b2 = np.zeros((1, y_scaled.shape[1]))

    n = len(x_scaled)
    for _ in range(epochs):
        # Обучение полностью локальное и повторяемое — без внешних сервисов и скрытой магии.
        z1 = x_scaled @ w1 + b1
        a1 = _relu(z1)
        y_pred = a1 @ w2 + b2

        error = y_pred - y_scaled
        grad_y = (2.0 / n) * error

        grad_w2 = a1.T @ grad_y
        grad_b2 = grad_y.sum(axis=0, keepdims=True)

        grad_a1 = grad_y @ w2.T
        grad_z1 = grad_a1 * _relu_grad(z1)
        grad_w1 = x_scaled.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)

        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1

    fitted_scaled = _relu(x_scaled @ w1 + b1) @ w2 + b2
    fitted = fitted_scaled * y_std + y_mean
    mse = float(np.mean((fitted - y) ** 2))
    var_y = float(np.var(y))
    r2 = 1.0 if var_y == 0 else 1.0 - mse / var_y

    return {
        "component_cols": component_cols,
        "window": window,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "train_mse": mse,
        "train_r2": r2,
    }


def _predict_next(model: dict, history_window: np.ndarray) -> np.ndarray:
    x = history_window.reshape(1, -1)
    x_scaled = (x - model["x_mean"]) / model["x_std"]
    pred_scaled = _relu(x_scaled @ model["w1"] + model["b1"]) @ model["w2"] + model["b2"]
    pred = pred_scaled * model["y_std"] + model["y_mean"]
    return np.clip(pred.reshape(-1), 0.0, 100.0)


def forecast_country_components(
    country_df: pd.DataFrame,
    model: dict,
    horizon: int,
) -> pd.DataFrame:
    component_cols = model["component_cols"]
    window = model["window"]

    base = country_df.sort_values("year").dropna(subset=component_cols).copy()
    if len(base) < window:
        raise ValueError("Not enough history for selected country.")

    history = base[component_cols].tail(window).to_numpy(dtype=float)
    last_year = int(base["year"].max())
    rows = []

    for step in range(1, horizon + 1):
        # Прогноз строим рекурсивно: предсказали следующий год, добавили его в историю и идём дальше.
        predicted = _predict_next(model, history)
        next_year = last_year + step
        row = {"year": next_year}
        for col, value in zip(component_cols, predicted):
            row[col] = float(value)
        rows.append(row)

        history = np.vstack([history[1:], predicted])

    return pd.DataFrame(rows)


def weighted_index_from_components(df: pd.DataFrame, weights: dict[str, float], component_cols: list[str]) -> pd.Series:
    return sum(df[col].fillna(0) * weights[col] for col in component_cols)
