from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # Для кластеризации по форме ряда нам важнее траектория, чем абсолютный уровень.
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


def _robust_scale_columns(
    matrix: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> np.ndarray:
    # Сначала слегка “прижимаем” выбросы, а уже потом кладём всё в общий диапазон.
    lower = np.quantile(matrix, lower_q, axis=0, keepdims=True)
    upper = np.quantile(matrix, upper_q, axis=0, keepdims=True)
    clipped = np.clip(matrix, lower, upper)
    span = upper - lower
    span = np.where(span == 0, 1.0, span)
    return (clipped - lower) / span


def prepare_convergence_matrix(
    df: pd.DataFrame,
    metric: str,
    year_from: int,
    year_to: int,
    min_coverage: float = 0.7,
) -> tuple[pd.DataFrame, list[int]]:
    # Здесь собираем ровную матрицу страна × годы для одного показателя.
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

    # Пропуски внутри ряда мягко интерполируем, чтобы не терять страну из-за одной-двух дырок.
    pivot = pivot.interpolate(axis=1, limit_direction="both")
    pivot = pivot.dropna()
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


def prepare_trend_feature_matrix(
    df: pd.DataFrame,
    metrics: list[str],
    year_from: int,
    year_to: int,
    min_coverage: float = 0.7,
) -> pd.DataFrame:
    # Во втором режиме превращаем каждый временной ряд в набор признаков: старт, конец, изменение, наклон и т.д.
    years = list(range(year_from, year_to + 1))
    if not metrics:
        return pd.DataFrame()

    base: pd.DataFrame | None = None
    year_vector = np.array(years, dtype=float)
    centered_years = year_vector - year_vector.mean()
    slope_denom = float((centered_years ** 2).sum()) or 1.0

    for metric in metrics:
        pivot = (
            df[df["year"].between(year_from, year_to)][["iso3", "year", metric]]
            .pivot(index="iso3", columns="year", values=metric)
            .reindex(columns=years)
        )

        coverage = pivot.notna().mean(axis=1)
        pivot = pivot.loc[coverage >= min_coverage].copy()
        if pivot.empty:
            return pd.DataFrame()

        pivot = pivot.interpolate(axis=1, limit_direction="both")
        pivot = pivot.dropna()
        if pivot.empty:
            return pd.DataFrame()

        values = pivot.to_numpy(dtype=float)
        diffs = np.diff(values, axis=1)
        start_values = values[:, 0]
        end_values = values[:, -1]
        changes = end_values - start_values
        centered_values = values - values.mean(axis=1, keepdims=True)
        slopes = (centered_values @ centered_years) / slope_denom
        diff_mean = diffs.mean(axis=1)
        diff_abs_mean = np.abs(diffs).mean(axis=1)
        diff_std = diffs.std(axis=1)
        increase_share = (diffs > 0).mean(axis=1)
        decrease_share = (diffs < 0).mean(axis=1)
        last_step = diffs[:, -1]

        metric_features = pd.DataFrame(
            {
                "iso3": pivot.index.astype(str),
                f"{metric}__start": start_values,
                f"{metric}__end": end_values,
                f"{metric}__change": changes,
                f"{metric}__slope": slopes,
                f"{metric}__diff_mean": diff_mean,
                f"{metric}__diff_abs_mean": diff_abs_mean,
                f"{metric}__diff_std": diff_std,
                f"{metric}__increase_share": increase_share,
                f"{metric}__decrease_share": decrease_share,
                f"{metric}__last_step": last_step,
            }
        )

        if base is None:
            base = metric_features
        else:
            base = base.merge(metric_features, on="iso3", how="inner")

    return pd.DataFrame() if base is None else base.reset_index(drop=True)


def add_activity_score(
    feature_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df

    activity_cols = [f"{metric}__diff_abs_mean" for metric in metrics if f"{metric}__diff_abs_mean" in feature_df.columns]
    if not activity_cols:
        out = feature_df.copy()
        out["activity_score"] = 0.0
        return out

    scaled = _robust_scale_columns(feature_df[activity_cols].to_numpy(dtype=float))
    activity_score = scaled.mean(axis=1)
    out = feature_df.copy()
    out["activity_score"] = activity_score
    return out


def robust_scale_feature_frame(
    feature_df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df

    if exclude_cols is None:
        exclude_cols = {"iso3", "club"}

    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    scaled_values = _robust_scale_columns(feature_df[feature_cols].to_numpy(dtype=float))

    out = feature_df.copy()
    out[feature_cols] = scaled_values
    return out


def run_kmeans_feature_space(
    feature_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 100,
) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df

    feature_cols = [col for col in feature_df.columns if col not in {"iso3", "club"}]
    values = feature_df[feature_cols].to_numpy(dtype=float)

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

    out = feature_df.copy()
    out["club"] = labels + 1
    return out


def summarize_clubs(clustered_df: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        long_df.groupby(["club", "year"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )

    members = (
        clustered_df[["iso3", "club"]]
        .sort_values(["club", "iso3"])
        .reset_index(drop=True)
    )
    return summary, members


def summarize_trend_clusters(
    clustered_df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clustered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        change_col = f"{metric}__change"
        slope_col = f"{metric}__slope"
        start_col = f"{metric}__start"
        end_col = f"{metric}__end"
        group = clustered_df.groupby("club", as_index=False)[[start_col, end_col, change_col, slope_col]].mean()
        for _, row in group.iterrows():
            rows.append(
                {
                    "club": int(row["club"]),
                    "metric": metric,
                    "mean_start": float(row[start_col]),
                    "mean_end": float(row[end_col]),
                    "mean_change": float(row[change_col]),
                    "mean_slope": float(row[slope_col]),
                }
            )

    summary = pd.DataFrame(rows)
    member_cols = ["iso3", "club"]
    if "activity_score" in clustered_df.columns:
        member_cols.append("activity_score")
    members = clustered_df[member_cols].sort_values(["club", "iso3"]).reset_index(drop=True)
    return summary, members


def project_feature_space_2d(feature_df: pd.DataFrame) -> pd.DataFrame:
    # Это не отдельная модель, а просто удобная проекция многомерных признаков на плоскость для визуализации.
    if feature_df.empty:
        return pd.DataFrame(columns=["iso3", "dim1", "dim2"])

    feature_cols = [col for col in feature_df.columns if col not in {"iso3", "club"}]
    values = feature_df[feature_cols].to_numpy(dtype=float)

    centered = values - values.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    return pd.DataFrame(
        {
            "iso3": feature_df["iso3"].astype(str).to_numpy(),
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
        }
    )


def hierarchical_cluster_merges(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=["left", "right", "distance", "size", "new_cluster"])

    feature_cols = [col for col in feature_df.columns if col not in {"iso3", "club"}]
    values = feature_df[feature_cols].to_numpy(dtype=float)
    labels = list(range(len(values)))
    clusters = {idx: [idx] for idx in labels}
    centroids = {idx: values[idx].copy() for idx in labels}
    sizes = {idx: 1 for idx in labels}
    next_cluster_id = len(values)
    merges: list[dict[str, float | int]] = []

    while len(clusters) > 1:
        active = sorted(clusters.keys())
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for i, left in enumerate(active[:-1]):
            for right in active[i + 1:]:
                distance = float(np.linalg.norm(centroids[left] - centroids[right]))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left, right)

        if best_pair is None:
            break

        left, right = best_pair
        new_size = sizes[left] + sizes[right]
        new_centroid = (
            centroids[left] * sizes[left] + centroids[right] * sizes[right]
        ) / new_size

        merges.append(
            {
                "left": left,
                "right": right,
                "distance": best_distance,
                "size": new_size,
                "new_cluster": next_cluster_id,
            }
        )

        clusters[next_cluster_id] = clusters[left] + clusters[right]
        centroids[next_cluster_id] = new_centroid
        sizes[next_cluster_id] = new_size

        del clusters[left]
        del clusters[right]
        del centroids[left]
        del centroids[right]
        del sizes[left]
        del sizes[right]

        next_cluster_id += 1

    return pd.DataFrame(merges)


def dendrogram_segments(
    merges_df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    if merges_df.empty:
        return pd.DataFrame(columns=["x", "y", "segment"])

    positions: dict[int, float] = {idx: float(idx) for idx in range(len(labels))}
    heights: dict[int, float] = {idx: 0.0 for idx in range(len(labels))}
    segments: list[dict[str, float | int]] = []
    segment_id = 0

    for _, row in merges_df.iterrows():
        left = int(row["left"])
        right = int(row["right"])
        height = float(row["distance"])
        new_cluster = int(row["new_cluster"])

        x_left = positions[left]
        x_right = positions[right]
        h_left = heights[left]
        h_right = heights[right]

        segments.extend(
            [
                {"x": x_left, "y": h_left, "segment": segment_id},
                {"x": x_left, "y": height, "segment": segment_id},
            ]
        )
        segment_id += 1
        segments.extend(
            [
                {"x": x_right, "y": h_right, "segment": segment_id},
                {"x": x_right, "y": height, "segment": segment_id},
            ]
        )
        segment_id += 1
        segments.extend(
            [
                {"x": x_left, "y": height, "segment": segment_id},
                {"x": x_right, "y": height, "segment": segment_id},
            ]
        )
        segment_id += 1

        positions[new_cluster] = (x_left + x_right) / 2.0
        heights[new_cluster] = height

    return pd.DataFrame(segments)
