from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # Для кластеризации важна форма траектории, а не абсолютный уровень.
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


def prepare_convergence_matrix(
    df: pd.DataFrame,
    metric: str,
    year_from: int,
    year_to: int,
    min_coverage: float = 0.7,
) -> tuple[pd.DataFrame, list[int]]:
    # Собираем ровную матрицу страна x годы для одного показателя.
    years = list(range(year_from, year_to + 1))
    pivot = (
        df[df["year"].between(year_from, year_to)][["iso3", "year", metric]]
        .pivot(index="iso3", columns="year", values=metric)
        .reindex(columns=years)
    )

    coverage = pivot.notna().mean(axis=1)
    pivot = pivot.loc[coverage >= min_coverage].copy()
    if pivot.empty:
        return pivot.reset_index(), years

    # Короткие пропуски заполняем интерполяцией, чтобы не терять страну из-за 1-2 дыр.
    pivot = pivot.interpolate(axis=1, limit_direction="both").dropna()
    return pivot.reset_index(), years


def run_kmeans(
    matrix_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 100,
) -> pd.DataFrame:
    if matrix_df.empty:
        return matrix_df

    values = matrix_df.drop(columns=["iso3"]).to_numpy(dtype=float)
    values = _normalize_rows(values)

    rng = np.random.default_rng(random_state)
    n_clusters = min(n_clusters, len(values))
    initial_idx = rng.choice(len(values), size=n_clusters, replace=False)
    centroids = values[initial_idx].copy()

    labels = np.zeros(len(values), dtype=int)
    for _ in range(max_iter):
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for cluster_id in range(n_clusters):
            cluster_points = values[labels == cluster_id]
            if len(cluster_points) == 0:
                centroids[cluster_id] = values[rng.integers(0, len(values))]
            else:
                centroids[cluster_id] = cluster_points.mean(axis=0)

    out = matrix_df.copy()
    out["club"] = labels + 1
    return out


def summarize_clubs(
    clustered_df: pd.DataFrame,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clustered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    long_df = clustered_df.melt(
        id_vars=["iso3", "club"],
        value_vars=years,
        var_name="year",
        value_name="value",
    )
    long_df["year"] = long_df["year"].astype(int)

    summary = (
        long_df.groupby(["club", "year"], as_index=False)
        .agg(mean_value=("value", "mean"))
        .sort_values(["club", "year"])
    )

    summary["start_value"] = summary.groupby("club")["mean_value"].transform("first")
    summary["mean_change"] = summary["mean_value"] - summary["start_value"]

    members = (
        clustered_df[["iso3", "club"]]
        .sort_values(["club", "iso3"])
        .reset_index(drop=True)
    )
    return summary, members
