import streamlit as st


# Единый словарь подписей нужен, чтобы все страницы называли метрики одинаково.
METRIC_LABELS = {
    "gdp_pc_usd": "ВВП на душу населения",
    "inflation_cpi": "Инфляция",
    "unemployment": "Безработица",
    "efi_total": "Индекс экономической свободы",
    "property_rights": "Права собственности",
    "government_integrity": "Честность правительства",
    "judicial_effectiveness": "Эффективность правосудия",
    "tax_burden": "Налоговая нагрузка",
    "government_spending": "Государственные расходы",
    "fiscal_health": "Фискальное здоровье",
    "business_freedom": "Свобода бизнеса",
    "labor_freedom": "Свобода труда",
    "monetary_freedom": "Денежная свобода",
    "trade_freedom": "Свобода торговли",
    "investment_freedom": "Свобода инвестиций",
    "financial_freedom": "Финансовая свобода",
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def apply_app_style() -> None:
    # Весь фирменный слой приложения собран здесь, чтобы страницы отвечали за аналитику, а не за CSS.
    st.markdown(
        """
        <style>
        :root {
            --rg-ink: #17231f;
            --rg-ink-soft: #4d625c;
            --rg-muted: #73857f;
            --rg-paper: #f6f3ee;
            --rg-paper-strong: rgba(255, 255, 255, 0.78);
            --rg-panel: rgba(255, 252, 247, 0.76);
            --rg-panel-strong: rgba(255, 255, 255, 0.88);
            --rg-stroke: rgba(23, 35, 31, 0.08);
            --rg-accent: #1f6b57;
            --rg-accent-soft: #d8ece4;
            --rg-warm: #efe4d8;
            --rg-shadow: 0 24px 60px rgba(28, 41, 37, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(205, 231, 219, 0.92), transparent 28%),
                radial-gradient(circle at 88% 0%, rgba(239, 228, 216, 0.85), transparent 22%),
                linear-gradient(180deg, #fbf8f3 0%, #f5f1ea 45%, #eef3ef 100%);
            color: var(--rg-ink);
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.7rem;
            padding-bottom: 2.4rem;
        }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(88, 129, 117, 0.26), transparent 26%),
                linear-gradient(180deg, #172821 0%, #22362f 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #f3efe6 !important;
        }
        section[data-testid="stSidebar"]::before {
            content: "RG";
            display: block;
            width: 2.65rem;
            height: 2.65rem;
            margin: 0 0 0.95rem 0.8rem;
            border-radius: 0.9rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.06));
            border: 1px solid rgba(255,255,255,0.14);
            color: #f7f2ea !important;
            font-family: "Avenir Next", "Helvetica Neue", sans-serif;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-align: center;
            line-height: 2.65rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            padding-top: 0.4rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 1rem;
            margin-bottom: 0.34rem;
            background: rgba(255,255,255,0.045);
            border: 1px solid transparent;
            transition: all 160ms ease;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: rgba(255,255,255,0.11);
            border-color: rgba(255,255,255,0.08);
            transform: translateX(2px);
        }
        h1, h2, h3 {
            color: #1f3a33 !important;
            letter-spacing: -0.02em;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif !important;
        }
        p, label, .stCaption, .stMarkdown, .stText {
            color: #314942;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        .stApp [data-baseweb="select"] *,
        .stApp [data-baseweb="input"] *,
        .stApp [data-baseweb="popover"] *,
        .stApp .stMultiSelect *,
        .stApp .stSelectbox *,
        .stApp .stNumberInput *,
        .stApp .stTextInput *,
        .stApp .stDateInput *,
        .stApp .stRadio *,
        .stApp .stCheckbox *,
        .stApp .stSlider *,
        .stApp .stTabs * {
            color: #1f3a33 !important;
        }
        .stApp .stAlert *,
        .stApp .stException *,
        .stApp .stInfo *,
        .stApp .stSuccess *,
        .stApp .stWarning *,
        .stApp .stError * {
            color: #1f3a33 !important;
        }
        .stApp input::placeholder,
        .stApp textarea::placeholder {
            color: #6d817a !important;
        }
        .stApp .stAlert,
        .stApp .stInfo,
        .stApp .stSuccess,
        .stApp .stWarning,
        .stApp .stError {
            border-radius: 1rem;
            border: 1px solid var(--rg-stroke);
            background: rgba(255,255,255,0.72);
        }
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stTable"] {
            background: rgba(255,255,255,0.68);
            border: 1px solid var(--rg-stroke);
            border-radius: 1.15rem;
            box-shadow: 0 10px 26px rgba(20, 34, 30, 0.05);
        }
        [data-testid="stMetricValue"] {
            color: #19332d !important;
        }
        [data-testid="stMetricLabel"] {
            color: #5a726a !important;
        }
        .rg-hero {
            padding: 1.65rem 1.75rem 1.7rem 1.75rem;
            border-radius: 2rem;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(247,243,236,0.84) 48%, rgba(239,246,241,0.82) 100%);
            border: 1px solid var(--rg-stroke);
            box-shadow: var(--rg-shadow);
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }
        .rg-hero::after {
            content: "";
            position: absolute;
            inset: auto -4rem -4rem auto;
            width: 14rem;
            height: 14rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(31,107,87,0.11) 0%, rgba(31,107,87,0) 70%);
            pointer-events: none;
        }
        .rg-brandline {
            display: inline-flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 1rem;
            padding: 0.45rem 0.7rem 0.45rem 0.45rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.6);
            border: 1px solid rgba(23,35,31,0.07);
            backdrop-filter: blur(10px);
        }
        .rg-logo {
            width: 2.7rem;
            height: 2.7rem;
            border-radius: 0.95rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(145deg, #17382f 0%, #235646 100%);
            color: #f9f4ec !important;
            font-weight: 800;
            letter-spacing: 0.14em;
            font-size: 0.95rem;
            box-shadow: 0 14px 28px rgba(23, 56, 47, 0.26);
        }
        .rg-brandtext {
            display: flex;
            flex-direction: column;
            gap: 0.08rem;
        }
        .rg-brandname {
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--rg-ink) !important;
            letter-spacing: 0.02em;
        }
        .rg-brandtag {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--rg-muted) !important;
        }
        .rg-hero-title {
            font-size: 2.55rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #17352e !important;
            max-width: 14ch;
            line-height: 0.96;
        }
        .rg-hero-text {
            font-size: 1rem;
            line-height: 1.72;
            color: #4f6760 !important;
            margin: 0;
            max-width: 58rem;
        }
        .rg-panel {
            padding: 1.15rem 1.15rem 1.2rem 1.15rem;
            border-radius: 1.5rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.84) 0%, rgba(249,247,241,0.72) 100%);
            border: 1px solid var(--rg-stroke);
            box-shadow: 0 14px 36px rgba(34, 58, 52, 0.05);
        }
        .rg-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #6a837b !important;
            margin-bottom: 0.5rem;
        }
        .rg-big {
            font-size: 1.65rem;
            font-weight: 800;
            color: #19332d !important;
            line-height: 1.06;
            margin-bottom: 0.42rem;
        }
        .rg-muted {
            color: #5b726c !important;
            line-height: 1.62;
            margin: 0;
        }
        .rg-note {
            padding: 0.95rem 1rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(31, 107, 87, 0.12);
            margin: 0.8rem 0 1rem 0;
            color: #35514a !important;
            box-shadow: 0 8px 24px rgba(29, 46, 41, 0.04);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stSlider {
            background: rgba(255, 255, 255, 0.72);
            border-radius: 1rem;
            border: 1px solid var(--rg-stroke);
        }
        .stDataFrame, .stPlotlyChart, iframe {
            border-radius: 1.2rem;
            overflow: hidden;
        }
        .stPlotlyChart {
            background: rgba(255,255,255,0.58);
            border: 1px solid var(--rg-stroke);
            box-shadow: 0 12px 30px rgba(20, 34, 30, 0.04);
            padding: 0.25rem;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(31, 107, 87, 0.18);
            background: linear-gradient(180deg, #fcfdfb 0%, #eef5f1 100%);
            color: #1f3a33 !important;
            font-weight: 600;
            padding: 0.46rem 1rem;
            box-shadow: 0 10px 20px rgba(20, 34, 30, 0.04);
        }
        .stButton > button:hover {
            border-color: rgba(31, 107, 87, 0.3);
            color: #17352e !important;
            transform: translateY(-1px);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.62);
            border: 1px solid var(--rg-stroke);
            border-radius: 1.2rem;
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 26px rgba(20, 34, 30, 0.04);
        }
        .stTabs [role="tablist"] {
            gap: 0.45rem;
        }
        .stTabs [role="tab"] {
            background: rgba(255,255,255,0.55);
            border: 1px solid var(--rg-stroke);
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(31,107,87,0.1);
            border-color: rgba(31,107,87,0.18);
        }
        .stExpander {
            border-radius: 1.15rem;
            border: 1px solid var(--rg-stroke);
            background: rgba(255,255,255,0.55);
        }
        @media (max-width: 900px) {
            .rg-hero-title {
                font-size: 2.05rem;
                max-width: none;
            }
            .rg-brandline {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="rg-hero">
            <div class="rg-brandline">
                <div class="rg-logo">RG</div>
                <div class="rg-brandtext">
                    <div class="rg-brandname">RegData</div>
                    <div class="rg-brandtag">Economic Signals, Clearly Mapped</div>
                </div>
            </div>
            <div class="rg-hero-title">{title}</div>
            <p class="rg-hero-text">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    st.markdown(f'<div class="rg-note">{text}</div>', unsafe_allow_html=True)


def render_panel(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="rg-panel">
            <div class="rg-kicker">{kicker}</div>
            <div class="rg-big">{title}</div>
            <p class="rg-muted">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
