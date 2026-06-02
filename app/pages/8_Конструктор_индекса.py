from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from regdata_core.analytics.neural_index import predict_neural_index, train_neural_index_model
from regdata_core.app_helpers import build_country_lookup, country_label
from regdata_core.data_processing.cache import COUNTRIES_PATH, EFI_PATH, load_parquet
from regdata_core.visualization.ui import apply_app_style, render_hero, render_note, metric_label


st.set_page_config(page_title="RegData — Конструктор индекса", layout="wide")
apply_app_style()
view_mode = st.radio("Режим просмотра", ["Базовый", "Продвинутый"], horizontal=True)

render_hero(
    "Конструктор индекса экономической свободы",
    "Выбирай веса для 12 компонент, считай свой вариант индекса и сравнивай его с официальным индексом Heritage.",
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

METHODOLOGY_TEXT = (
    "Официальный индекс Heritage строится на 12 компонентах, сгруппированных в четыре блока: "
    "верховенство права, размер правительства, эффективность регулирования и открытость рынков. "
    "В базовой методологии все 12 компонент имеют одинаковый вес, а итоговый индекс рассчитывается "
    "как их среднее значение."
)


def calculate_weighted_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    # Индекс считаем только по тем компонентам, которые реально есть в строке, чтобы пропуски не превращались в ложные нули.
    weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
    available_weight = pd.Series(0.0, index=df.index, dtype=float)

    for component in EFI_COMPONENTS:
        values = pd.to_numeric(df[component], errors="coerce")
        has_value = values.notna().astype(float)
        weighted_sum = weighted_sum + values.fillna(0.0) * weights[component]
        available_weight = available_weight + has_value * weights[component]

    return weighted_sum / available_weight.where(available_weight > 0)


def build_weighted_feature_frame(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    # Для нейросети подаём те же веса прямо во входные признаки, чтобы обучение и сравнение были согласованы.
    weighted = df.copy()
    for component in EFI_COMPONENTS:
        weighted[component] = weighted[component] * weights[component] * len(EFI_COMPONENTS)
    return weighted


def finite_weight_sensitivity(row: pd.Series, weights: dict[str, float], epsilon: float = 0.01) -> pd.DataFrame:
    # Это маленький “what if”: слегка увеличиваем один вес и смотрим, как дрогнет итоговый индекс.
    base_index = calculate_weighted_index(pd.DataFrame([row]), weights).iloc[0]
    rows: list[dict[str, float | str]] = []

    for component in EFI_COMPONENTS:
        bumped = weights.copy()
        bumped[component] += epsilon
        total = sum(bumped.values())
        bumped = {key: value / total for key, value in bumped.items()}
        bumped_index = calculate_weighted_index(pd.DataFrame([row]), bumped).iloc[0]
        rows.append(
            {
                "component": component,
                "weight_share": weights[component],
                "base_value": float(row[component]),
                "sensitivity": float(bumped_index - base_index),
            }
        )

    return pd.DataFrame(rows)


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
def train_model_cached(
    path_str: str,
    feature_cols: tuple[str, ...],
    weight_signature: tuple[tuple[str, float], ...],
) -> dict:
    df = load_parquet(Path(path_str))
    weights = dict(weight_signature)
    weighted_df = build_weighted_feature_frame(df, weights)
    return train_neural_index_model(weighted_df, list(feature_cols))


if not EFI_PATH.exists():
    st.warning("Сначала обнови индекс экономической свободы на странице карты.")
    st.stop()

efi = load_efi_cached(str(EFI_PATH))
missing_components = [col for col in EFI_COMPONENTS if col not in efi.columns]
if missing_components:
    st.warning("EFI загружен в старом формате без компонент. Нажми «Обновить индекс экономической свободы» на странице карты.")
    st.stop()

countries = load_countries_cached(str(COUNTRIES_PATH))
country_lookup = build_country_lookup(countries)

render_note(METHODOLOGY_TEXT)

years = sorted(efi["year"].dropna().astype(int).unique())
default_year = years[-1]

col1, col2 = st.columns([1.2, 1])
with col1:
    year = st.slider("Год", int(min(years)), int(max(years)), int(default_year))
with col2:
    iso_options = sorted(efi["iso3"].dropna().astype(str).unique(), key=lambda iso: country_label(iso, country_lookup))
    country_iso = st.selectbox(
        "Страна для детального просмотра",
        iso_options,
        format_func=lambda iso: country_label(iso, country_lookup),
    )

st.subheader("Настройка весов")
render_note("Ты задаёшь важность каждой компоненты. Ниже веса автоматически нормируются, поэтому их удобно воспринимать как относительную значимость.")

weight_cols = st.columns(3)
weights = {}
for idx, component in enumerate(EFI_COMPONENTS):
    with weight_cols[idx % 3]:
        weights[component] = st.slider(metric_label(component), 0, 100, 10, 1, key=f"builder_{component}")

total_weight = sum(weights.values())
if total_weight == 0:
    st.warning("Нужно задать хотя бы один ненулевой вес.")
    st.stop()

norm_weights = {key: value / total_weight for key, value in weights.items()}
weight_signature = tuple((component, round(norm_weights[component], 8)) for component in EFI_COMPONENTS)
# Модель переобучаем под текущую конфигурацию весов, чтобы сравнение было честным именно для этой формулы.
model = train_model_cached(str(EFI_PATH), tuple(EFI_COMPONENTS), weight_signature)

render_note(
    "Экспериментальная локальная нейросетевая модель переобучается под выбранные веса. "
    "Это делает сравнение более последовательным и понятным."
)

df_year = efi[efi["year"] == year].copy()
df_year["custom_index"] = calculate_weighted_index(df_year, norm_weights)
weighted_features = build_weighted_feature_frame(df_year[EFI_COMPONENTS], norm_weights)
df_year["neural_predicted_index"] = predict_neural_index(model, weighted_features)
df_year = df_year.dropna(subset=["efi_total", "custom_index", "neural_predicted_index"])
df_year["official_rank"] = df_year["efi_total"].rank(ascending=False, method="min")
df_year["custom_rank"] = df_year["custom_index"].rank(ascending=False, method="min")
df_year["neural_rank"] = df_year["neural_predicted_index"].rank(ascending=False, method="min")
df_year["rank_diff"] = df_year["custom_rank"] - df_year["official_rank"]
df_year["Страна"] = df_year["iso3"].map(lambda iso: country_label(iso, country_lookup))

if df_year.empty:
    st.warning("Нет данных для выбранного года.")
    st.stop()

correlation = df_year[["efi_total", "custom_index"]].corr().iloc[0, 1]
mae = (df_year["efi_total"] - df_year["custom_index"]).abs().mean()
neural_corr = df_year[["efi_total", "neural_predicted_index"]].corr().iloc[0, 1]
country_row = df_year[df_year["iso3"] == country_iso]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Корреляция с официальным индексом", f"{correlation:.3f}")
with c2:
    st.metric("Среднее отклонение", f"{mae:.2f}")
with c3:
    st.metric("Нейросетевая корреляция", f"{neural_corr:.3f}")

render_note(
    f"Качество локальной модели на исторических данных: R² = {model['train_r2']:.4f}, "
    f"MSE = {model['train_mse']:.4f}."
)
if not country_row.empty:
    row_summary = country_row.iloc[0]
    direction = "выше" if row_summary["rank_diff"] < 0 else "ниже" if row_summary["rank_diff"] > 0 else "на том же уровне"
    render_note(
        f"Для {country_label(country_iso, country_lookup)} твоя конфигурация ставит страну {direction} официального места. "
        "Если хочешь понять причину, смотри веса и чувствительность компонент."
    )
    render_note(
        "Краткий вывод для защиты: изменение весов компонент действительно меняет положение стран в рейтинге, "
        "а значит структура индекса влияет не только на значения, но и на итоговую интерпретацию институционального качества."
    )

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Кастомный индекс vs официальный")
    render_note(
        "Чем ближе точки к воображаемой диагонали, тем ближе твой индекс к официальному. Сильные отклонения означают заметное изменение ранжирования."
    )
    fig = px.scatter(
        df_year,
        x="efi_total",
        y="custom_index",
        hover_name="Страна",
        color="rank_diff",
        color_continuous_scale=["#b85c4c", "#f2efe9", "#1f6b57"],
        hover_data={
            "official_rank": True,
            "custom_rank": True,
            "rank_diff": True,
            "efi_total": ":.2f",
            "custom_index": ":.2f",
        },
    )
    fig.update_layout(
        xaxis_title="Официальный индекс Heritage",
        yaxis_title="Кастомный индекс",
        plot_bgcolor="rgba(255,255,255,0.74)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar_title="Разница мест",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
    fig.update_traces(
        marker=dict(size=11, opacity=0.85, line=dict(color="white", width=1.2)),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Страна в фокусе")
    if country_row.empty:
        render_note("Для выбранной страны нет полного набора компонент в этом году.")
    else:
        row = country_row.iloc[0]
        points_diff = float(row["custom_index"] - row["efi_total"])
        st.metric("Официальный индекс", f"{row['efi_total']:.2f}", f"место: {int(row['official_rank'])}")
        # Здесь специально показываем разницу в пунктах сразу в карточке — так эффект от движения слайдеров заметнее.
        st.metric("Кастомный индекс", f"{row['custom_index']:.2f}", f"{points_diff:+.2f} пункта")
        st.metric("Нейросеточный прогноз", f"{row['neural_predicted_index']:.2f}", f"место: {int(row['neural_rank'])}")
        st.metric("Разница мест", f"{int(row['rank_diff']):+d}")
        render_note(
            f"Для выбранной страны текущие веса меняют итог на {points_diff:+.2f} пункта "
            f"и сдвигают её на {int(row['rank_diff']):+d} позиций относительно официального места."
        )

        comp_df = pd.DataFrame(
            {
                "Компонента": [metric_label(comp) for comp in EFI_COMPONENTS],
                "Значение": [row[comp] for comp in EFI_COMPONENTS],
                "Вес": [norm_weights[comp] * 100 for comp in EFI_COMPONENTS],
            }
        )
        fig_comp = px.bar(
            comp_df,
            x="Вес",
            y="Компонента",
            orientation="h",
            color="Значение",
            color_continuous_scale=["#d8e2dc", "#a3b18a", "#588157", "#344e41"],
        )
        fig_comp.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            plot_bgcolor="rgba(255,255,255,0.74)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_comp.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
        st.plotly_chart(fig_comp, use_container_width=True)

        if view_mode == "Продвинутый":
            st.subheader("Чувствительность индекса к весам")
            render_note(
                "Здесь показано, какие компоненты сильнее всего двигают итоговый индекс этой страны, если немного увеличить их вес."
            )
            sensitivity_df = finite_weight_sensitivity(row, norm_weights)
            sensitivity_df["Компонента"] = sensitivity_df["component"].map(metric_label)
            fig_sens = px.bar(
                sensitivity_df.sort_values("sensitivity", ascending=False),
                x="sensitivity",
                y="Компонента",
                orientation="h",
                color="sensitivity",
                color_continuous_scale=["#d8ece4", "#7fb3a3", "#1f6b57"],
            )
            fig_sens.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
                plot_bgcolor="rgba(255,255,255,0.74)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Изменение индекса при небольшом увеличении веса",
            )
            fig_sens.update_xaxes(showgrid=True, gridcolor="rgba(31, 58, 51, 0.10)", zeroline=False)
            st.plotly_chart(fig_sens, use_container_width=True)

st.subheader("Нормированные веса")
weights_df = pd.DataFrame(
    {
        "Компонента": [metric_label(comp) for comp in EFI_COMPONENTS],
        "Вес, %": [round(norm_weights[comp] * 100, 2) for comp in EFI_COMPONENTS],
    }
).sort_values("Вес, %", ascending=False)
if view_mode == "Базовый":
    render_note("В базовом режиме показаны только самые значимые веса. Полная структура и сравнение по всем странам доступны в продвинутом режиме.")
    st.dataframe(weights_df.head(5), use_container_width=True, hide_index=True)
else:
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

if view_mode == "Продвинутый":
    with st.expander("Показать таблицу совпадения по странам"):
        export_compare_df = (
            df_year[["Страна", "efi_total", "custom_index", "neural_predicted_index", "official_rank", "custom_rank", "neural_rank", "rank_diff"]]
            .rename(
                columns={
                    "efi_total": "Официальный индекс",
                    "custom_index": "Кастомный индекс",
                    "neural_predicted_index": "Нейросеточный прогноз",
                    "official_rank": "Официальное место",
                    "custom_rank": "Кастомное место",
                    "neural_rank": "Нейросеточное место",
                    "rank_diff": "Разница мест",
                }
            )
            .sort_values("Кастомное место")
            .reset_index(drop=True)
        )
        st.download_button(
            "Скачать сравнение индексов CSV",
            data=export_compare_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"regdata_index_builder_{year}.csv",
            mime="text/csv",
        )
        st.dataframe(export_compare_df, use_container_width=True)
