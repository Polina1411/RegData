from __future__ import annotations

import numpy as np
import pandas as pd


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def train_neural_index_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "efi_total",
    hidden_size: int = 16,
    epochs: int = 1200,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> dict:
    # Это компактная локальная нейросеть: она учится восстанавливать итоговый EFI по его компонентам.
    train_df = df[feature_cols + [target_col]].dropna().copy()
    if train_df.empty:
        raise ValueError("No training data available for neural model.")

    x = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df[[target_col]].to_numpy(dtype=float)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std == 0, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std = np.where(y_std == 0, 1.0, y_std)
    y_scaled = (y - y_mean) / y_std

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.2, size=(x_scaled.shape[1], hidden_size))
    b1 = np.zeros((1, hidden_size))
    w2 = rng.normal(0, 0.2, size=(hidden_size, 1))
    b2 = np.zeros((1, 1))

    n = len(x_scaled)
    for _ in range(epochs):
        # Здесь самый обычный цикл обучения: прямой проход, ошибка и обновление весов.
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
        "feature_cols": feature_cols,
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


def predict_neural_index(model: dict, feature_df: pd.DataFrame) -> np.ndarray:
    # На предсказании повторяем ту же нормализацию, что и на обучении.
    x = feature_df[model["feature_cols"]].to_numpy(dtype=float)
    x_scaled = (x - model["x_mean"]) / model["x_std"]
    pred_scaled = _relu(x_scaled @ model["w1"] + model["b1"]) @ model["w2"] + model["b2"]
    pred = pred_scaled * model["y_std"] + model["y_mean"]
    return pred.reshape(-1)
