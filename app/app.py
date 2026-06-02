from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"

# Центральная точка навигации: здесь просто собираются все страницы приложения.
navigation = st.navigation(
    [
        st.Page(APP_DIR / "home.py", title="Главная", icon=":material/home:"),
        st.Page(PAGES_DIR / "1_Обзор.py", title="Обзор", icon=":material/insights:"),
        st.Page(PAGES_DIR / "2_Карта.py", title="Карта", icon=":material/public:"),
        st.Page(PAGES_DIR / "3_Страна.py", title="Страна", icon=":material/flag:"),
        st.Page(PAGES_DIR / "4_Корреляция.py", title="Корреляция", icon=":material/scatter_plot:"),
        st.Page(PAGES_DIR / "5_Временные_ряды.py", title="Временные ряды", icon=":material/show_chart:"),
        st.Page(PAGES_DIR / "6_Экономическая_свобода.py", title="Экономическая свобода", icon=":material/trending_up:"),
        st.Page(PAGES_DIR / "7_Клубы_конвергенции.py", title="Клубы конвергенции", icon=":material/hub:"),
        st.Page(PAGES_DIR / "8_Конструктор_индекса.py", title="Конструктор индекса", icon=":material/tune:"),
        st.Page(PAGES_DIR / "9_Прогноз_индекса.py", title="Прогноз индекса", icon=":material/auto_graph:"),
    ]
)

# Дальше Streamlit уже сам открывает выбранную страницу.
navigation.run()
