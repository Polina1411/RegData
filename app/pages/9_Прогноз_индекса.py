from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from regdata_core.analytics.efi_forecast import (
    forecast_country_components,
    train_component_forecaster,
)
from regdata_core.analytics.convergence import (
    add_activity_score,
    dendrogram_segments,
    hierarchical_cluster_merges,
    prepare_trend_feature_matrix,
    project_feature_space_2d,
    robust_scale_feature_frame,
)
from regdata_core.app_helpers import build_country_lookup, country_label
from regdata_core.data_processing.cache import COUNTRIES_PATH, EFI_PATH, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Прогноз индекса", layout="wide")
apply_app_style()
view_mode = st.radio("Режим просмотра", ["Базовый", "Продвинутый"], horizontal=True)

render_hero(
    "Прогноз индекса экономической свободы",
    "Локальная нейросеть прогнозирует будущие значения 12 компонент EFI по истории страны, "
    "а затем рассчитывает индекс по выбранным тобой весам.",
)

EFI_COMPONENTS = [
    "property_rights",
    "government_integrity",
    "judicial_effectiveness",
    "tax_burden",
    "government_spending",
    "fiscal_health",
    "business_freedom",
    "labor_freedom",
    "monetary_freedom",
    "trade_freedom",
    "investment_freedom",
    "financial_freedom",
]


def calculate_weighted_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    # Логика та же, что и в конструкторе: считаем только по доступным компонентам, без штрафа за пропуски как за нули.
    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    available_weight = pd.Series(0.0, index=df.index, dtype=float)

    for component in EFI_COMPONENTS:
        values = pd.to_numeric(df[component], errors="coerce")
        has_value = values.notna().astype(float)
        weighted_sum = weighted_sum + values.fillna(0.0) * weights[component]
        available_weight = available_weight + has_value * weights[component]

    return weighted_sum / available_weight.where(available_weight > 0)


def nearest_countries_by_efi_dynamics(
    efi_df: pd.DataFrame,
    country_iso: str,
    years_back: int = 10,
    limit: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Похожие страны ищем не по уровню EFI, а по форме недавней динамики его компонент.
    max_year = int(efi_df["year"].dropna().max())
    year_from = max(int(efi_df["year"].dropna().min()), max_year - years_back + 1)
    trend_df = prepare_trend_feature_matrix(efi_df, EFI_COMPONENTS, year_from, max_year, 0.7)
    if trend_df.empty or country_iso not in trend_df["iso3"].values:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trend_df = add_activity_score(trend_df, EFI_COMPONENTS)
    scaled_df = robust_scale_feature_frame(trend_df)
    feature_cols = [col for col in scaled_df.columns if col != "iso3"]
    target = scaled_df.loc[scaled_df["iso3"] == country_iso, feature_cols].iloc[0].to_numpy(dtype=float)

    distances = []
    for _, row in scaled_df.iterrows():
        values = row[feature_cols].to_numpy(dtype=float)
        distances.append(float(np.linalg.norm(values - target)))

    scored = scaled_df.copy()
    scored["distance"] = distances
    scored = scored.sort_values("distance").head(limit).reset_index(drop=True)
    projected = project_feature_space_2d(scored)
    merges = hierarchical_cluster_merges(scored)
    return scored, projected, merges


@st.cache_data
def load_efi_cached(path_str: str) -> pd.DataFrame:
    return load_parquet(Path(path_str))


@st.cache_data
def load_countries_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "country"])
    return load_parquet(path)


@st.cache_data
def train_forecaster_cached(path_str: str) -> dict:
    df = load_parquet(Path(path_str))
    return train_component_forecaster(df, EFI_COMPONENTS, window=3)


if not EFI_PATH.exists():
    st.warning("Сначала обнови индекс экономической свободы на странице карты.")
    st.stop()

efi = load_efi_cached(str(EFI_PATH))
countries = load_countries_cached(str(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)
model = train_forecaster_cached(str(EFI_PATH))

render_note(
    "Как это работает: модель обучается на исторических рядах 12 компонент индекса, "
    "затем прогнозирует будущие компоненты, а итоговый индекс считается по выбранным тобой весам."
)
render_note(
    f"Качество модели на обучающей выборке: R² = {model['train_r2']:.4f}, MSE = {model['train_mse']:.4f}."
)

iso_options = sorted(efi["iso3"].dropna().astype(str).unique(), key=lambda iso: country_label(iso, country_lookup))

col1, col2 = st.columns([1.2, 1])
with col1:
    country_iso = st.selectbox(
        "Страна",
        iso_options,
        format_func=lambda iso: country_label(iso, country_lookup),
    )
with col2:
    horizon = st.slider("Горизонт прогноза, лет", 1, 5, 3)

st.subheader("Веса компонент")
weight_cols = st.columns(3)
weights = {}
for idx, component in enumerate(EFI_COMPONENTS):
    with weight_cols[idx % 3]:
        weights[component] = st.slider(f"{metric_label(component)} ", 0, 100, 10, 1, key=f"forecast_{component}")

total_weight = sum(weights.values())
if total_weight == 0:
    st.warning("Нужно задать хотя бы один ненулевой вес.")
    st.stop()

norm_weights = {key: value / total_weight for key, value in weights.items()}

country_df = efi[efi["iso3"] == country_iso].sort_values("year").copy()
if len(country_df.dropna(subset=EFI_COMPONENTS)) < 3:
    st.warning("Для выбранной страны недостаточно истории для прогноза.")
    st.stop()

forecast_df = forecast_country_components(country_df, model, horizon=horizon)
# Сразу считаем два сценария: базовый с равными весами и пользовательский.
forecast_df["custom_index"] = calculate_weighted_index(forecast_df, norm_weights)
forecast_df["baseline_index"] = calculate_weighted_index(
    forecast_df,
    {component: 1 / len(EFI_COMPONENTS) for component in EFI_COMPONENTS},
)

actual_df = country_df[["year", "efi_total"] + EFI_COMPONENTS].dropna(subset=["efi_total"]).copy()
actual_df["custom_index"] = calculate_weighted_index(actual_df, norm_weights)
actual_df["baseline_index"] = calculate_weighted_index(
    actual_df,
    {component: 1 / len(EFI_COMPONENTS) for component in EFI_COMPONENTS},
)
actual_df["official_index"] = actual_df["efi_total"]
actual_df["Тип"] = "Факт"

forecast_df["official_index"] = pd.NA
forecast_df["Тип"] = "Прогноз"

combined = pd.concat(
    [
        actual_df[["year", "official_index", "baseline_index", "custom_index", "Тип"]],
        forecast_df[["year", "official_index", "baseline_index", "custom_index", "Тип"]],
    ],
    ignore_index=True,
)

country_name = country_label(country_iso, country_lookup)

c1, c2, c3 = st.columns(3)
with c1:
    last_actual = actual_df.iloc[-1]
    st.metric("Последний официальный индекс", f"{last_actual['efi_total']:.2f}", f"{int(last_actual['year'])}")
with c2:
    first_pred = forecast_df.iloc[0]
    st.metric("Первый прогнозный год", f"{first_pred['custom_index']:.2f}", f"{int(first_pred['year'])}")
with c3:
    last_pred = forecast_df.iloc[-1]
    st.metric("Конец горизонта", f"{last_pred['custom_index']:.2f}", f"{int(last_pred['year'])}")
trend_direction = "рост" if last_pred["custom_index"] > last_actual["efi_total"] else "снижение"
render_note(
    f"Главный вывод: для {country_name} модель сейчас показывает {trend_direction} индекса на выбранном горизонте. "
    "Сравни базовый ряд и ряд по твоим весам, чтобы понять, насколько выбор весов меняет траекторию."
)

st.subheader(country_name)
render_note(
    "Сплошные линии показывают фактические значения, пунктирные — прогноз. Точки помогают увидеть, где заканчивается история и начинается модельный расчёт."
)
plot_df = combined.melt(
    id_vars=["year", "Тип"],
    value_vars=["official_index", "baseline_index", "custom_index"],
    var_name="series",
    value_name="value",
)
plot_df["Ряд"] = plot_df["series"].map(
    {
        "official_index": "Официальный индекс Heritage",
        "baseline_index": "Базовый индекс (равные веса)",
        "custom_index": "Индекс по твоим весам",
    }
)

fig = px.line(
    plot_df,
    x="year",
    y="value",
    color="Ряд",
    line_dash="Тип",
    markers=True,
    color_discrete_sequence=["#2f5d50", "#8d99ae", "#d17b49"],
)
fig.update_layout(
    xaxis_title="Год",
    yaxis_title="Индекс экономической свободы",
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

similar_df, projected_similar_df, merges_df = nearest_countries_by_efi_dynamics(efi, country_iso)
if not similar_df.empty and view_mode == "Продвинутый":
    similar_df["Страна"] = similar_df["iso3"].map(lambda iso: country_label(iso, country_lookup))
    projected_similar_df = projected_similar_df.merge(
        similar_df[["iso3", "Страна", "distance", "activity_score"]],
        on="iso3",
        how="left",
    )

    st.subheader("Похожие страны по динамике EFI")
    render_note(
        "Здесь сравнение идёт не по текущему уровню индекса, а по форме динамики компонент за последние годы."
    )

    s_left, s_right = st.columns([1.1, 1])
    with s_left:
        fig_sim = px.scatter(
            projected_similar_df,
            x="dim1",
            y="dim2",
            color="distance",
            hover_name="Страна",
            size="activity_score",
            size_max=22,
            color_continuous_scale=["#d8ece4", "#7fb3a3", "#1f6b57"],
            hover_data={"distance": ":.3f", "activity_score": ":.2f", "dim1": False, "dim2": False},
        )
        fig_sim.update_layout(
            xaxis_title="Ось сходства 1",
            yaxis_title="Ось сходства 2",
            plot_bgcolor="rgba(255,255,255,0.74)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar_title="Дистанция",
        )
        fig_sim.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
        fig_sim.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
        fig_sim.update_traces(marker=dict(opacity=0.88, line=dict(color="white", width=1.2)))
        st.plotly_chart(fig_sim, use_container_width=True)

    with s_right:
        st.dataframe(
            similar_df[["iso3", "distance", "activity_score"]]
            .assign(Страна=lambda data: data["iso3"].map(lambda iso: country_label(iso, country_lookup)))
            .rename(columns={"iso3": "ISO3", "distance": "Дистанция", "activity_score": "Подвижность"})
            [["Страна", "ISO3", "Дистанция", "Подвижность"]]
            .round(3),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Иерархическое дерево похожих стран")
    if not merges_df.empty:
        dendro_labels = [
            country_label(iso, country_lookup).replace(f" ({iso})", "")
            for iso in similar_df["iso3"].tolist()
        ]
        segment_df = dendrogram_segments(merges_df, dendro_labels)
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
            xaxis_title="Похожие страны",
            yaxis_title="Расстояние по EFI-динамике",
            plot_bgcolor="rgba(255,255,255,0.74)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=480,
        )
        dendro_fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(dendro_labels))),
            ticktext=dendro_labels,
            showgrid=False,
            tickangle=45,
        )
        dendro_fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
        st.plotly_chart(dendro_fig, use_container_width=True)

weights_df = pd.DataFrame(
    {
        "Компонента": [metric_label(comp) for comp in EFI_COMPONENTS],
        "Вес, %": [round(norm_weights[comp] * 100, 2) for comp in EFI_COMPONENTS],
    }
).sort_values("Вес, %", ascending=False)

if view_mode == "Базовый":
    st.subheader("Краткий прогноз")
    render_note("Базовый режим оставляет только итоговую траекторию индекса. Детальный прогноз 12 компонент и полные веса вынесены в продвинутый режим.")
    compact_table = forecast_df[["year", "baseline_index", "custom_index"]].rename(
        columns={
            "year": "Год",
            "baseline_index": "Базовый индекс",
            "custom_index": "Индекс по твоим весам",
        }
    )
    st.dataframe(compact_table.round(2), use_container_width=True, hide_index=True)
else:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Прогноз компонент")
        component_table = forecast_df[["year"] + EFI_COMPONENTS].rename(
            columns={"year": "Год", **{col: metric_label(col) for col in EFI_COMPONENTS}}
        )
        st.dataframe(component_table.round(2), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Нормированные веса")
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

if view_mode == "Продвинутый":
    with st.expander("Показать фактические и прогнозные значения индекса"):
        table_df = combined[["year", "Тип", "official_index", "baseline_index", "custom_index"]].rename(
            columns={
                "year": "Год",
                "official_index": "Официальный индекс Heritage",
                "baseline_index": "Базовый индекс (равные веса)",
                "custom_index": "Индекс по твоим весам",
            }
        )
        st.download_button(
            "Скачать прогноз CSV",
            data=table_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"regdata_forecast_{country_iso}.csv",
            mime="text/csv",
        )
        st.dataframe(table_df.round(2), use_container_width=True, hide_index=True)
