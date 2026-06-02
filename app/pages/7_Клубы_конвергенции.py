import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

from regdata_core.analytics.convergence import (
    add_activity_score,
    dendrogram_segments,
    hierarchical_cluster_merges,
    prepare_convergence_matrix,
    prepare_trend_feature_matrix,
    project_feature_space_2d,
    robust_scale_feature_frame,
    run_kmeans,
    run_kmeans_feature_space,
    summarize_clubs,
    summarize_trend_clusters,
)
from regdata_core.app_helpers import (
    build_analytics_dataset,
    build_country_lookup,
    country_label,
    current_year,
    metrics_for_df,
)
from regdata_core.data_processing.cache import WDI_PATH, COUNTRIES_PATH, OECD_RECENT_PATH, load_parquet
from regdata_core.data_processing.cache import LIGHT_COUNTRIES_GEOJSON_PATH, RAW_COUNTRIES_GEOJSON_PATH
from regdata_core.data_processing.geojson import build_lightweight_geojson
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Клубы конвергенции", layout="wide")
apply_app_style()
view_mode = st.radio("Режим просмотра", ["Базовый", "Продвинутый"], horizontal=True)

render_hero(
    "Клубы конвергенции",
    "Разделение стран на группы с похожими траекториями. "
    "Можно кластеризовать либо один временной ряд, либо направление изменений сразу по нескольким показателям.",
)


@st.cache_data
def load_wdi_cached(path_str: str) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


@st.cache_data
def load_geojson_cached(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_lightweight_geojson() -> Path | None:
    if LIGHT_COUNTRIES_GEOJSON_PATH.exists():
        return LIGHT_COUNTRIES_GEOJSON_PATH
    if not RAW_COUNTRIES_GEOJSON_PATH.exists():
        return None
    build_lightweight_geojson(
        source_path=RAW_COUNTRIES_GEOJSON_PATH,
        output_path=LIGHT_COUNTRIES_GEOJSON_PATH,
        iso_key="ISO3166-1-Alpha-3",
        decimals=1,
    )
    return LIGHT_COUNTRIES_GEOJSON_PATH


def build_cluster_labels(summary_df: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    for club_id, club_df in summary_df.groupby("club"):
        # Названия клубов делаем не формальными, а по смыслу доминирующих изменений.
        growth = club_df.sort_values("mean_change", ascending=False).iloc[0]
        decline = club_df.sort_values("mean_change", ascending=True).iloc[0]

        parts = []
        if growth["mean_change"] > 0:
            parts.append(f"рост {metric_label(str(growth['metric'])).lower()}")
        if decline["mean_change"] < 0:
            parts.append(f"снижение {metric_label(str(decline['metric'])).lower()}")

        if not parts:
            parts.append("смешанная динамика")

        labels[int(club_id)] = " / ".join(parts)
    return labels


def activity_band(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Нет оценки"
    value = float(score)
    if value < 0.2:
        return "Почти без изменений"
    if value < 0.45:
        return "Слабые изменения"
    if value < 0.7:
        return "Умеренные изменения"
    return "Сильные изменения"


def build_cluster_centroid_frame(clustered_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in clustered_df.columns if col not in {"iso3", "club"}]
    return clustered_df.groupby("club", as_index=False)[feature_cols].mean()


def short_single_mode_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "Недостаточно данных для вывода."
    latest = summary_df.sort_values("year").groupby("Клуб", as_index=False).tail(1)
    leader = latest.sort_values("mean_value", ascending=False).iloc[0]["Клуб"]
    lagger = latest.sort_values("mean_value", ascending=True).iloc[0]["Клуб"]
    return (
        f"На текущем горизонте выше остальных выглядит {leader}, а слабее остальных — {lagger}. "
        "Главное здесь не только уровень, но и то, насколько траектории расходятся между собой."
    )


def short_multi_mode_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "Недостаточно данных для вывода."
    club_changes = summary_df.groupby("Клуб", as_index=False)["mean_change"].mean()
    leader = club_changes.sort_values("mean_change", ascending=False).iloc[0]["Клуб"]
    lagger = club_changes.sort_values("mean_change", ascending=True).iloc[0]["Клуб"]
    return (
        f"Сейчас самый сильный набор положительных изменений у группы «{leader}», "
        f"а самый слабый или отрицательный — у группы «{lagger}»."
    )


if not WDI_PATH.exists():
    st.warning("Данные WDI не найдены. Сначала обнови данные на странице карты.")
    st.stop()

wdi = load_wdi_cached(str(WDI_PATH))
wdi = build_analytics_dataset(
    wdi,
    load_parquet(OECD_RECENT_PATH) if OECD_RECENT_PATH.exists() else pd.DataFrame(),
)
countries = load_countries_cached(str(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)
metrics = metrics_for_df(wdi)
mode = st.radio(
    "Режим кластеризации",
    [
        "По траектории одного показателя",
        "По росту и снижению по нескольким показателям",
    ],
    horizontal=True,
)

year_values = wdi["year"].dropna().astype(int)
min_year = int(year_values.min())
max_year = int(year_values.max()) if not year_values.empty else current_year()

col1, col2, col3 = st.columns([1.2, 1.5, 1])

selected_metrics: list[str] = []
metric = metrics[0]
with col1:
    if mode == "По траектории одного показателя":
        metric = st.selectbox("Показатель", metrics, format_func=metric_label)
    else:
        selected_metrics = st.multiselect(
            "Показатели",
            metrics,
            default=metrics[: min(3, len(metrics))],
            format_func=metric_label,
        )
with col2:
    year_from, year_to = st.slider("Период анализа", min_year, max_year, (max(min_year, 2010), max_year))
with col3:
    n_clusters = st.slider("Количество клубов", 2, 6, 3)

coverage = st.slider("Минимальная полнота ряда", 0.4, 1.0, 0.7, 0.05)

if mode == "По траектории одного показателя":
    matrix_df, years = prepare_convergence_matrix(
        wdi,
        metric=metric,
        year_from=year_from,
        year_to=year_to,
        min_coverage=coverage,
    )

    if matrix_df.empty:
        st.warning("Недостаточно данных для построения клубов конвергенции в выбранном диапазоне.")
        st.stop()

    clustered_df = run_kmeans(matrix_df, n_clusters=n_clusters)
    summary_df, members_df = summarize_clubs(clustered_df, years)
    members_df["Страна"] = members_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
    members_df["Клуб"] = members_df["club"].map(lambda x: f"Клуб {x}")
    summary_df["Клуб"] = summary_df["club"].map(lambda x: f"Клуб {x}")

    render_note(
        "Клубы формируются по форме одного временного ряда. "
        "Так можно увидеть страны с похожей долгосрочной траекторией по выбранному показателю."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Стран в анализе", int(len(clustered_df)))
    with c2:
        st.metric("Клубов", int(clustered_df["club"].nunique()))
    with c3:
        st.metric("Период", f"{year_from}–{year_to}")

    st.subheader("Средние траектории клубов")
    render_note(
        "Здесь показаны средние траектории по каждому клубу. Смотри на форму линий и расхождение между ними, а не только на конечный уровень."
    )
    fig = px.line(
        summary_df,
        x="year",
        y="mean_value",
        color="Клуб",
        markers=True,
        color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
    )
    fig.update_layout(
        xaxis_title="Год",
        yaxis_title=metric_label(metric),
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig.update_traces(
        line=dict(width=3.3, shape="spline", smoothing=0.55),
        marker=dict(size=7, line=dict(color="white", width=1)),
    )
    st.plotly_chart(fig, use_container_width=True)
    render_note(short_single_mode_summary(summary_df))

    st.subheader("Размеры клубов")
    club_sizes = members_df.groupby("Клуб", as_index=False).size().rename(columns={"size": "Стран"})
    fig_sizes = px.bar(
        club_sizes,
        x="Клуб",
        y="Стран",
        color="Клуб",
        color_discrete_sequence=["#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9", "#b7b7a4"],
    )
    fig_sizes.update_layout(
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

    st.subheader("Состав клубов")
    if view_mode == "Базовый":
        render_note("В базовом режиме показаны только первые страны из клубов, чтобы экран оставался компактным.")
        st.dataframe(
            members_df[["Клуб", "Страна", "iso3"]].rename(columns={"iso3": "ISO3"}).head(15),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            members_df[["Клуб", "Страна", "iso3"]].rename(columns={"iso3": "ISO3"}),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Показать матрицу стран и траекторий"):
            display_df = clustered_df.copy()
            display_df["Страна"] = display_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
            display_df["Клуб"] = display_df["club"].map(lambda x: f"Клуб {x}")
            st.dataframe(
                display_df.rename(columns={"iso3": "ISO3"})[["Клуб", "Страна", "ISO3"] + years],
                use_container_width=True,
                hide_index=True,
            )
else:
    if len(selected_metrics) < 2:
        st.warning("Выбери хотя бы два показателя для кластеризации по росту и снижению.")
        st.stop()

    trend_df = prepare_trend_feature_matrix(
        wdi,
        metrics=selected_metrics,
        year_from=year_from,
        year_to=year_to,
        min_coverage=coverage,
    )

    if trend_df.empty:
        st.warning("Недостаточно данных для кластеризации по выбранным показателям и периоду.")
        st.stop()

    trend_df = add_activity_score(trend_df, selected_metrics)
    scaled_trend_df = robust_scale_feature_frame(trend_df)
    # Во втором режиме кластеризуем уже не сырые значения, а набор признаков по динамике нескольких показателей.
    clustered_df = run_kmeans_feature_space(scaled_trend_df, n_clusters=n_clusters)
    summary_df, members_df = summarize_trend_clusters(clustered_df, selected_metrics)
    cluster_labels = build_cluster_labels(summary_df)
    summary_df["Клуб"] = summary_df["club"].map(lambda x: f"Клуб {x}: {cluster_labels.get(int(x), 'динамика')}")
    summary_df["Показатель"] = summary_df["metric"].map(metric_label)
    members_df["Страна"] = members_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
    members_df["Клуб"] = members_df["club"].map(lambda x: f"Клуб {x}: {cluster_labels.get(int(x), 'динамика')}")
    members_df["Активность"] = members_df["activity_score"].map(activity_band) if "activity_score" in members_df.columns else "Нет оценки"
    projected_df = project_feature_space_2d(clustered_df).merge(
        members_df[["iso3", "Страна", "Клуб", "club", "activity_score", "Активность"]],
        on="iso3",
        how="left",
    )
    projected_df["Размер точки"] = 10 + projected_df["activity_score"].fillna(0) * 14
    centroid_df = build_cluster_centroid_frame(clustered_df)
    centroid_df["iso3"] = centroid_df["club"].map(lambda x: f"Клуб {int(x)}")
    merges_df = hierarchical_cluster_merges(centroid_df.drop(columns=["club"]))
    dendro_labels = [
        f"Клуб {int(club_id)}"
        for club_id in centroid_df["club"].tolist()
    ]
    segment_df = dendrogram_segments(merges_df, dendro_labels)

    render_note(
        "В этом режиме страны группируются по направлению и силе изменений сразу по нескольким показателям. "
        "То есть в один клуб попадают страны с похожим набором ростов и снижений."
    )
    render_note(short_multi_mode_summary(summary_df))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Стран в анализе", int(len(clustered_df)))
    with c2:
        st.metric("Клубов", int(clustered_df["club"].nunique()))
    with c3:
        st.metric("Показателей", int(len(selected_metrics)))
    with c4:
        st.metric("Период", f"{year_from}–{year_to}")

    if "activity_score" in members_df.columns:
        activity_mean = float(members_df["activity_score"].mean())
        activity_max_country = members_df.sort_values("activity_score", ascending=False).iloc[0]
        extra1, extra2 = st.columns(2)
        with extra1:
            st.metric("Средняя активность изменений", f"{activity_mean:.2f}")
        with extra2:
            st.metric("Самая подвижная страна", activity_max_country["Страна"])

    render_note(
        "Краткий вывод для защиты: кластеризация разделяет страны не по одному уровню показателя, "
        "а по сходству траекторий и направлению изменений. Это позволяет увидеть устойчивые типы экономической динамики."
    )

    st.subheader("Среднее изменение по клубам")
    render_note(
        "Положительные столбцы означают средний рост показателя внутри клуба, отрицательные — среднее снижение."
    )
    fig_change = px.bar(
        summary_df,
        x="Показатель",
        y="mean_change",
        color="Клуб",
        barmode="group",
        color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
        hover_data={"mean_slope": ":.3f", "mean_start": ":.2f", "mean_end": ":.2f"},
    )
    fig_change.update_layout(
        xaxis_title="Показатель",
        yaxis_title="Среднее изменение за период",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_change.update_xaxes(showgrid=False)
    fig_change.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=True, zerolinecolor="rgba(31, 58, 51, 0.16)")
    st.plotly_chart(fig_change, use_container_width=True)

    st.subheader("2D-проекция кластеров")
    render_note(
        "Каждая точка — страна. Чем ближе точки друг к другу, тем более похожа их общая динамика по выбранным показателям."
    )
    fig_scatter = px.scatter(
        projected_df,
        x="dim1",
        y="dim2",
        color="Клуб",
        hover_name="Страна",
        size="Размер точки",
        size_max=22,
        color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
        hover_data={
            "dim1": False,
            "dim2": False,
            "club": False,
            "activity_score": ":.2f",
            "Активность": True,
            "Размер точки": False,
        },
    )
    fig_scatter.update_layout(
        xaxis_title="Ось сходства 1",
        yaxis_title="Ось сходства 2",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
        height=540,
    )
    fig_scatter.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig_scatter.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig_scatter.update_traces(marker=dict(opacity=0.88, line=dict(color="white", width=1.2)))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Средний наклон тренда")
    slope_df = summary_df.copy()
    fig_slope = px.bar(
        slope_df,
        x="Показатель",
        y="mean_slope",
        color="Клуб",
        barmode="group",
        color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
    )
    fig_slope.update_layout(
        xaxis_title="Показатель",
        yaxis_title="Средний наклон в год",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    fig_slope.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=True, zerolinecolor="rgba(31, 58, 51, 0.16)")
    st.plotly_chart(fig_slope, use_container_width=True)

    if view_mode == "Продвинутый":
        st.subheader("Иерархическое дерево клубов")
        render_note(
            "Здесь дерево строится не по всем странам сразу, а по центрам найденных клубов. "
            "Чем ниже соединение, тем ближе друг к другу целые группы стран по своей динамике."
        )
        if segment_df.empty:
            render_note("Для выбранного набора стран и показателей дерево сейчас построить не удалось.")
        else:
            dendro_fig = go.Figure()
            for segment_id, seg in segment_df.groupby("segment"):
                dendro_fig.add_trace(
                    go.Scatter(
                        x=seg["x"],
                        y=seg["y"],
                        mode="lines",
                        line=dict(color="#355c53", width=1.8),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            dendro_fig.update_layout(
                xaxis_title="Клубы",
                yaxis_title="Расстояние между центрами клубов",
                plot_bgcolor="rgba(255,255,255,0.74)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=420,
            )
            dendro_fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(dendro_labels))),
                ticktext=dendro_labels,
                showgrid=False,
                tickangle=0,
            )
            dendro_fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
            st.plotly_chart(dendro_fig, use_container_width=True)

    st.subheader("Карта кластеров")
    geojson_path = ensure_lightweight_geojson()
    if geojson_path is None:
        render_note("Файл геометрии стран не найден, поэтому карта кластеров сейчас недоступна.")
    else:
        geojson = load_geojson_cached(str(geojson_path))
        map_df = members_df[["iso3", "Клуб", "club", "Страна"]].copy()
        map_df["club_code"] = map_df["club"].astype(int)
        fig_map = px.choropleth(
            map_df,
            geojson=geojson,
            locations="iso3",
            featureidkey="properties.iso3",
            color="Клуб",
            hover_name="Страна",
            hover_data={"iso3": True, "club": False, "club_code": False},
            color_discrete_sequence=["#2f3e46", "#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9"],
            projection="natural earth",
        )
        fig_map.update_geos(
            fitbounds="locations",
            showcountries=True,
            countrycolor="rgba(255,255,255,0.75)",
            showcoastlines=False,
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            legend_title_text="",
            height=560,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Размеры клубов")
    club_sizes = members_df.groupby("Клуб", as_index=False).size().rename(columns={"size": "Стран"})
    fig_sizes = px.bar(
        club_sizes,
        x="Клуб",
        y="Стран",
        color="Клуб",
        color_discrete_sequence=["#52796f", "#84a98c", "#a4c3b2", "#cad2c5", "#ddbea9", "#b7b7a4"],
    )
    fig_sizes.update_layout(
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sizes, use_container_width=True)

    st.subheader("Состав клубов")
    export_members_df = members_df[["Клуб", "Страна", "iso3", "Активность"]].rename(columns={"iso3": "ISO3"})
    st.download_button(
        "Скачать состав клубов CSV",
        data=export_members_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="regdata_trend_clusters.csv",
        mime="text/csv",
    )
    if view_mode == "Базовый":
        render_note("В базовом режиме показана сокращённая выборка стран. Полный состав и технические таблицы доступны в продвинутом режиме.")
        st.dataframe(
            export_members_df.head(15),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            export_members_df,
            use_container_width=True,
            hide_index=True,
        )

    if view_mode == "Продвинутый":
        with st.expander("Показать таблицу средних изменений по клубам"):
            display_summary = summary_df[
                ["Клуб", "Показатель", "mean_start", "mean_end", "mean_change", "mean_slope"]
            ].rename(
                columns={
                    "mean_start": "Среднее значение в начале",
                    "mean_end": "Среднее значение в конце",
                    "mean_change": "Среднее изменение",
                    "mean_slope": "Средний наклон в год",
                }
            )
            st.dataframe(display_summary.round(3), use_container_width=True, hide_index=True)

        with st.expander("Показать числовые признаки для кластеризации"):
            feature_view = clustered_df.copy()
            feature_view["Страна"] = feature_view["iso3"].map(lambda iso: country_label(iso, country_lookup))
            feature_view["Клуб"] = feature_view["club"].map(lambda x: f"Клуб {x}: {cluster_labels.get(int(x), 'динамика')}")
            keep_cols = ["Клуб", "Страна", "iso3", "activity_score"]
            metric_cols = [col for col in feature_view.columns if "__change" in col or "__slope" in col or "__diff_abs_mean" in col]
            st.dataframe(
                feature_view[keep_cols + metric_cols].rename(columns={"iso3": "ISO3", "activity_score": "Индекс активности"}).round(3),
                use_container_width=True,
                hide_index=True,
            )
