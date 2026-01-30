import streamlit as st
import pandas as pd

from regdata_core.data_processing.cache import WDI_PATH, load_parquet

st.set_page_config(page_title="RegData — Scatter", layout="wide")
st.title("📊 Scatter (WDI): связь показателей по годам")

# ---------- Load data ----------
if not WDI_PATH.exists():
    st.warning("WDI кэш не найден. Перейди на страницу Map и нажми «Скачать/обновить WDI».")
    st.stop()

@st.cache_data
def load_wdi():
    return load_parquet(WDI_PATH)

wdi = load_wdi()

# ---------- Controls ----------
st.subheader("Параметры")

metrics = ["gdp_pc_usd", "inflation_cpi", "unemployment"]

# страна: если выбрана на карте — подставляем её как дефолт
iso_candidates = sorted(wdi["iso3"].dropna().unique())
default_iso3 = st.session_state.get("selected_iso3")
if default_iso3 not in iso_candidates:
    default_iso3 = iso_candidates[0]

col1, col2, col3 = st.columns(3)

with col1:
    iso3 = st.selectbox(
        "Страна (ISO3)",
        iso_candidates,
        index=iso_candidates.index(default_iso3),
    )

with col2:
    x_metric = st.selectbox("X-показатель", metrics, index=0)

with col3:
    y_metric = st.selectbox("Y-показатель", metrics, index=2 if "unemployment" in metrics else 1)

# опция: ограничение по годам
min_year = int(wdi["year"].dropna().min())
max_year = int(wdi["year"].dropna().max())
year_from, year_to = st.slider("Период", min_year, max_year, (min_year, max_year))

# ---------- Filter ----------
df = wdi[(wdi["iso3"] == iso3) & (wdi["year"].between(year_from, year_to))][["year", x_metric, y_metric]].dropna()
df = df.sort_values("year")

if df.empty:
    st.warning("Нет данных для выбранной комбинации (страна/метрики/период).")
    st.stop()

# ---------- Scatter ----------
st.subheader(f"Scatter: {x_metric} vs {y_metric} ({iso3})")

# Streamlit scatter_chart expects index as x-axis.
# It plots the columns against index; for scatter we want (x_metric, y_metric).
# Поэтому используем st.scatter_chart на датафрейме с двумя колонками.
plot_df = df[[x_metric, y_metric]].copy()
st.scatter_chart(plot_df, use_container_width=True)

# ---------- Correlation + info ----------
corr = df[[x_metric, y_metric]].corr().iloc[0, 1]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Наблюдений (лет)", int(len(df)))
with c2:
    st.metric("Корреляция Пирсона", f"{corr:.3f}")
with c3:
    st.metric("Период", f"{int(df['year'].min())}–{int(df['year'].max())}")

# ---------- Table ----------
with st.expander("Показать таблицу данных"):
    st.dataframe(df, use_container_width=True)

st.caption("Подсказка: выбери страну на Map — она автоматически подставится тут.")
