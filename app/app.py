import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="RegData", layout="wide")

st.title("RegData — Economic Freedom Visualizer (MVP)")
st.caption("MVP: каркас приложения + демо-визуализации. Далее подключим реальные данные (EFI/WDI).")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Параметры")
    year = st.slider("Год", 2000, 2024, 2019)
    metric = st.selectbox("Показатель", ["EFI (demo)", "GDP per capita (demo)", "Inflation (demo)"])

with col2:
    st.subheader("Демо-данные (заглушка)")
    np.random.seed(42)
    countries = ["USA", "GBR", "DEU", "FRA", "CHN", "IND", "BRA", "RUS", "JPN", "CAN"]
    df = pd.DataFrame({
        "country": countries,
        "value": np.random.normal(70, 10, size=len(countries)).clip(0, 100),
        "year": year,
        "metric": metric
    })
    st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("Простейшая визуализация (MVP-график)")

fig = px.bar(df, x="country", y="value", title=f"{metric} — {year}")
st.plotly_chart(fig, use_container_width=True)

st.info("Следующий шаг: подключаем реальный датасет и делаем страницы: Карта / Страна / Корреляции / Временные ряды.")
