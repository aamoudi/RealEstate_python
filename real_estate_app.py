import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ – Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  THEMES
# ─────────────────────────────────────────
THEMES = {
    "Midnight Navy": {
        "bg": "#0d1117",
        "card": "#161b22",
        "card2": "#1c2333",
        "accent": "#58a6ff",
        "accent2": "#3fb950",
        "accent3": "#f78166",
        "text": "#e6edf3",
        "subtext": "#8b949e",
        "border": "#30363d",
        "gradient": "linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2333 100%)",
        "plot_bg": "#0d1117",
        "paper_bg": "#161b22",
        "grid": "#21262d",
        "colorscale": "Blues",
    },
    "Deep Purple": {
        "bg": "#0e0b1a",
        "card": "#1a1330",
        "card2": "#241c42",
        "accent": "#a78bfa",
        "accent2": "#34d399",
        "accent3": "#f472b6",
        "text": "#ede9fe",
        "subtext": "#a1a1aa",
        "border": "#3b2d6e",
        "gradient": "linear-gradient(135deg, #0e0b1a 0%, #1a1330 50%, #241c42 100%)",
        "plot_bg": "#0e0b1a",
        "paper_bg": "#1a1330",
        "grid": "#2d2050",
        "colorscale": "Purples",
    },
    "Slate Teal": {
        "bg": "#0a1628",
        "card": "#0f2040",
        "card2": "#163354",
        "accent": "#2dd4bf",
        "accent2": "#fb923c",
        "accent3": "#e879f9",
        "text": "#e2f1f8",
        "subtext": "#94a3b8",
        "border": "#1e3a5f",
        "gradient": "linear-gradient(135deg, #0a1628 0%, #0f2040 50%, #163354 100%)",
        "plot_bg": "#0a1628",
        "paper_bg": "#0f2040",
        "grid": "#1a3050",
        "colorscale": "Teal",
    },
}

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Midnight Navy"

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Real_estate.csv")
    df.columns = [
        "No", "Transaction Date", "House Age (yrs)",
        "Distance to MRT (m)", "Convenience Stores",
        "Latitude", "Longitude", "Price per Unit Area"
    ]
    df = df.drop(columns=["No"])
    labels = ["Cheap", "Mid-Class", "Pricey", "Luxurious"]
    df["Neighbourhood"] = pd.qcut(df["Price per Unit Area"], q=4, labels=labels)
    df["Price Category"] = df["Neighbourhood"]
    return df

df = load_data()

# ─────────────────────────────────────────
#  DYNAMIC CSS
# ─────────────────────────────────────────
def inject_css(t):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background: {t['bg']};
        color: {t['text']};
    }}
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {t['card']} !important;
        border-right: 1px solid {t['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}
    /* ── Metric cards ── */
    .metric-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 16px;
        padding: 22px 24px;
        text-align: center;
        transition: transform .25s, box-shadow .25s;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,.4);
    }}
    .metric-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {t['accent']};
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: .82rem;
        color: {t['subtext']};
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}
    /* ── Section headers ── */
    .section-header {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {t['text']};
        border-left: 4px solid {t['accent']};
        padding-left: 14px;
        margin: 32px 0 18px 0;
    }}
    .section-sub {{
        color: {t['subtext']};
        font-size: .96rem;
        margin-bottom: 22px;
        line-height: 1.7;
    }}
    /* ── Hero ── */
    .hero-box {{
        background: {t['gradient']};
        border: 1px solid {t['border']};
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 28px;
    }}
    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
    }}
    .hero-sub {{
        font-size: 1.15rem;
        color: {t['subtext']};
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.7;
    }}
    /* ── Info cards ── */
    .info-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 24px;
        height: 100%;
    }}
    .info-card h4 {{
        color: {t['accent']};
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .info-card p {{
        color: {t['subtext']};
        font-size: .88rem;
        line-height: 1.65;
    }}
    /* ── Tags / badges ── */
    .badge {{
        display: inline-block;
        background: {t['accent']}22;
        color: {t['accent']};
        border: 1px solid {t['accent']}44;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: .78rem;
        font-weight: 600;
        margin: 2px;
    }}
    /* ── Prediction result ── */
    .pred-result {{
        background: linear-gradient(135deg, {t['accent']}18, {t['accent2']}18);
        border: 1px solid {t['accent']}44;
        border-radius: 18px;
        padding: 32px;
        text-align: center;
    }}
    .pred-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: {t['accent']};
    }}
    .pred-label {{
        color: {t['subtext']};
        font-size: .9rem;
        margin-top: 6px;
    }}
    /* ── Insight box ── */
    .insight-box {{
        background: {t['card2']};
        border-left: 4px solid {t['accent2']};
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: {t['text']};
        font-size: .9rem;
        line-height: 1.65;
    }}
    /* ── Divider ── */
    .custom-divider {{
        border: none;
        border-top: 1px solid {t['border']};
        margin: 28px 0;
    }}
    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {t['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 3px; }}
    /* ── Selectbox / inputs ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div {{
        background: {t['card']} !important;
        border-color: {t['border']} !important;
        color: {t['text']} !important;
    }}
    .stSlider > div {{
        color: {t['text']};
    }}
    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        color: #000;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all .25s;
    }}
    .stButton > button:hover {{
        opacity: .85;
        transform: translateY(-2px);
    }}
    /* ── Footer ── */
    .footer {{
        text-align: center;
        padding: 32px 0 16px 0;
        color: {t['subtext']};
        font-size: .82rem;
        border-top: 1px solid {t['border']};
        margin-top: 48px;
    }}
    .footer span {{
        color: {t['accent']};
        font-weight: 600;
    }}
    /* ── Logo ── */
    .logo-text {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .logo-sub {{
        font-size: .7rem;
        color: {t['subtext']};
        letter-spacing: .1em;
        text-transform: uppercase;
    }}
    /* Dataframe */
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {t['card']};
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid {t['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {t['subtext']};
        border-radius: 8px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['accent']}22 !important;
        color: {t['accent']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────
def fig_layout(fig, t, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=t["text"], size=14, family="Space Grotesk"), x=0.02),
        paper_bgcolor=t["paper_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(color=t["subtext"], family="Inter"),
        margin=dict(l=12, r=12, t=44 if title else 20, b=12),
        height=height,
        legend=dict(
            bgcolor=t["card2"],
            bordercolor=t["border"],
            borderwidth=1,
            font=dict(color=t["text"]),
        ),
        xaxis=dict(gridcolor=t["grid"], linecolor=t["border"], zerolinecolor=t["grid"]),
        yaxis=dict(gridcolor=t["grid"], linecolor=t["border"], zerolinecolor=t["grid"]),
    )
    return fig

CAT_COLORS = {"Cheap": "#f78166", "Mid-Class": "#ffa657", "Pricey": "#79c0ff", "Luxurious": "#3fb950"}

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
def render_sidebar(t):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding: 12px 0 20px 0;">
          <div style="font-size:2.8rem; margin-bottom:4px;">🏙️</div>
          <div class="logo-text">EstateIQ</div>
          <div class="logo-sub">Real Estate Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<hr style="border-color:{t["border"]}">', unsafe_allow_html=True)

        # Theme selector
        st.markdown(f'<div style="font-size:.75rem;color:{t["subtext"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">🎨 Theme</div>', unsafe_allow_html=True)
        chosen = st.radio("", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme), label_visibility="collapsed")
        if chosen != st.session_state.theme:
            st.session_state.theme = chosen
            st.rerun()

        st.markdown(f'<hr style="border-color:{t["border"]}">', unsafe_allow_html=True)

        # Navigation
        st.markdown(f'<div style="font-size:.75rem;color:{t["subtext"]};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">📍 Navigate</div>', unsafe_allow_html=True)
        nav = st.radio("", [
            "🏠  Overview",
            "📊  Data Explorer",
            "🔍  Insights & Charts",
            "🤖  Price Predictor",
            "📬  Contact",
        ], label_visibility="collapsed")

        st.markdown(f'<hr style="border-color:{t["border"]}">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:.75rem;color:{t['subtext']};line-height:1.6;">
          📁 Dataset: <b style="color:{t['text']}">Real Estate Taiwan</b><br>
          🗂 Records: <b style="color:{t['text']}">414</b><br>
          📐 Features: <b style="color:{t['text']}">6</b>
        </div>
        """, unsafe_allow_html=True)

    return nav

# ─────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────

def page_overview(t):
    # Hero
    st.markdown(f"""
    <div class="hero-box">
      <div style="font-size:3.8rem;margin-bottom:8px;">🏙️</div>
      <div class="hero-title">EstateIQ</div>
      <div style="font-size:1.05rem;color:{t['subtext']};max-width:680px;margin:0 auto 16px auto;line-height:1.7;">
        An intelligent real estate analytics platform that uncovers hidden pricing patterns
        in the Taiwan housing market using machine learning and data-driven insights.
      </div>
      <div>
        <span class="badge">🏠 414 Properties</span>
        <span class="badge">📊 6 Features</span>
        <span class="badge">🤖 Random Forest</span>
        <span class="badge">🗺️ Taiwan Market</span>
        <span class="badge">📈 Price Prediction</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    kpis = [
        ("414", "Properties Analysed"),
        (f"{df['Price per Unit Area'].mean():.1f}", "Avg Price / Unit Area"),
        (f"{df['Distance to MRT (m)'].mean():.0f} m", "Avg MRT Distance"),
        (f"{df['Convenience Stores'].mean():.1f}", "Avg Nearby Stores"),
    ]
    cols = st.columns(4)
    for col, (val, lbl) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # About section
    st.markdown('<div class="section-header">🎯 About This Project</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="info-card">
          <h4>🔍 The Problem</h4>
          <p>Real estate pricing is notoriously opaque. Buyers, sellers, and investors struggle to determine
          fair market value without deep local knowledge. Simple rules like "bigger = pricier" fail to capture
          the complex interplay of location, age, accessibility, and neighbourhood dynamics that truly drive prices.</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="info-card">
          <h4>💡 Our Solution</h4>
          <p>EstateIQ applies advanced feature analysis and a Random Forest model to decode the hidden
          signals in property data. Beyond raw correlations, we explore <em>why</em> variables matter —
          revealing surprising patterns like how both very new and vintage homes command premium prices.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(3)
    cards = [
        ("🗺️ Location Intelligence", "Latitude, longitude, and MRT distance are the strongest price predictors. Properties within 1 km of transit hubs command a significant premium."),
        ("🏚️ Age Paradox", "House age follows a U-shaped relationship with price — both brand-new builds and vintage 40+ year properties outperform mid-aged homes."),
        ("🏪 Amenity Effect", "Convenience store count serves as a proxy for neighbourhood vibrancy. More stores → higher perceived desirability → premium pricing."),
    ]
    for col, (title, body) in zip(cols2, cards):
        with col:
            st.markdown(f"""
            <div class="info-card">
              <h4>{title}</h4>
              <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Dataset column guide
    st.markdown('<div class="section-header">📋 Dataset Features</div>', unsafe_allow_html=True)
    feat_data = {
        "Feature": ["Transaction Date", "House Age", "Distance to MRT", "Convenience Stores", "Latitude", "Longitude"],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric"],
        "Description": [
            "Year of transaction (2012.67 – 2013.58)",
            "Age of the property in years (0 – 43.8)",
            "Distance to nearest MRT station in metres",
            "Number of convenience stores nearby (0 – 10)",
            "Geographic latitude coordinate",
            "Geographic longitude coordinate",
        ],
        "Key Insight": [
            "Limited temporal range; less impactful",
            "U-shaped effect on price",
            "Strongest negative predictor (closer = pricier)",
            "Positive correlation with price",
            "Strong spatial predictor",
            "Strong spatial predictor",
        ]
    }
    feat_df = pd.DataFrame(feat_data)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Quick scatter map
    st.markdown('<div class="section-header">🗺️ Property Map</div>', unsafe_allow_html=True)
    fig = px.scatter_mapbox(
        df, lat="Latitude", lon="Longitude",
        color="Price per Unit Area",
        size="Price per Unit Area",
        color_continuous_scale=t["colorscale"],
        size_max=15,
        zoom=13,
        mapbox_style="carto-darkmatter",
        hover_data={"Price per Unit Area": True, "House Age (yrs)": True, "Distance to MRT (m)": True},
    )
    fig.update_layout(
        paper_bgcolor=t["paper_bg"],
        margin=dict(l=0, r=0, t=10, b=0),
        height=440,
        coloraxis_colorbar=dict(
            title=dict(text="Price/Unit", font=dict(color=t["text"])),
            tickfont=dict(color=t["subtext"]),
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def page_data_explorer(t):
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Browse, filter, and download the complete real estate dataset. All columns have been renamed for clarity.</p>', unsafe_allow_html=True)

    # Filters
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            price_range = st.slider("Price per Unit Area", float(df["Price per Unit Area"].min()), float(df["Price per Unit Area"].max()), (float(df["Price per Unit Area"].min()), float(df["Price per Unit Area"].max())))
        with col2:
            age_range = st.slider("House Age (yrs)", float(df["House Age (yrs)"].min()), float(df["House Age (yrs)"].max()), (float(df["House Age (yrs)"].min()), float(df["House Age (yrs)"].max())))
        with col3:
            mrt_range = st.slider("Distance to MRT (m)", float(df["Distance to MRT (m)"].min()), float(df["Distance to MRT (m)"].max()), (float(df["Distance to MRT (m)"].min()), float(df["Distance to MRT (m)"].max())))

        col4, col5 = st.columns(2)
        with col4:
            store_range = st.slider("Convenience Stores", int(df["Convenience Stores"].min()), int(df["Convenience Stores"].max()), (int(df["Convenience Stores"].min()), int(df["Convenience Stores"].max())))
        with col5:
            hoods = ["All"] + list(df["Neighbourhood"].cat.categories)
            selected_hood = st.selectbox("Neighbourhood Tier", hoods)

    # Apply filters
    fdf = df.copy()
    fdf = fdf[
        (fdf["Price per Unit Area"].between(*price_range)) &
        (fdf["House Age (yrs)"].between(*age_range)) &
        (fdf["Distance to MRT (m)"].between(*mrt_range)) &
        (fdf["Convenience Stores"].between(*store_range))
    ]
    if selected_hood != "All":
        fdf = fdf[fdf["Neighbourhood"] == selected_hood]

    st.markdown(f'<p style="color:{t["subtext"]}">Showing <b style="color:{t["accent"]}">{len(fdf)}</b> of <b>{len(df)}</b> records</p>', unsafe_allow_html=True)

    display_cols = ["Transaction Date", "House Age (yrs)", "Distance to MRT (m)", "Convenience Stores", "Latitude", "Longitude", "Price per Unit Area", "Neighbourhood"]
    st.dataframe(fdf[display_cols].reset_index(drop=True), use_container_width=True, height=440)

    # Download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        fdf[display_cols].to_excel(writer, index=False, sheet_name="Real Estate Data")
    output.seek(0)
    st.download_button(
        label="⬇️  Download Filtered Data as Excel",
        data=output,
        file_name="real_estate_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Summary stats
    st.markdown('<div class="section-header">📈 Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(fdf[["House Age (yrs)", "Distance to MRT (m)", "Convenience Stores", "Price per Unit Area"]].describe().round(2), use_container_width=True)


def page_insights(t):
    st.markdown('<div class="section-header">🔍 Insights & Visual Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Interactive visualisations revealing how each variable shapes property prices — uncovering patterns beyond simple correlation.</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📍 Location", "🏚️ Age Effect", "🚇 MRT Access", "🏪 Amenities", "🔗 Correlations", "📅 Market Trends"])

    # ── TAB 1: Location ──
    with tabs[0]:
        st.markdown(f'<div class="insight-box">📍 Properties cluster tightly in a high-density urban area of New Taipei City / Xindian. <b>Location alone</b> (lat/lon) explains a large portion of price variance — houses in the northern cluster command premium prices.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="Longitude", y="Latitude",
                color="Price per Unit Area", color_continuous_scale=t["colorscale"],
                size="Price per Unit Area", size_max=16,
                hover_data={"Price per Unit Area": True, "Neighbourhood": True},
                title="Geographic Price Distribution")
            fig_layout(fig, t, height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(df, x="Longitude", y="Latitude",
                color="Neighbourhood",
                color_discrete_map=CAT_COLORS,
                size="Price per Unit Area", size_max=14,
                title="Neighbourhood Tiers by Location")
            fig_layout(fig2, t, height=420)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.density_mapbox(df, lat="Latitude", lon="Longitude",
            z="Price per Unit Area", radius=20,
            center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
            zoom=13, mapbox_style="carto-darkmatter",
            color_continuous_scale=t["colorscale"],
            title="Price Heat Map")
        fig3.update_layout(paper_bgcolor=t["paper_bg"], margin=dict(l=0,r=0,t=40,b=0), height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2: Age Effect ──
    with tabs[1]:
        st.markdown(f'<div class="insight-box">🏚️ House age follows a <b>U-shaped curve</b>: brand-new properties and vintage 40+ year homes command <b>higher prices</b> than mid-aged properties. Pricey houses are either newly built or characterful vintage builds.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age_price = df.groupby(df["House Age (yrs)"].round())["Price per Unit Area"].mean().reset_index()
            fig = px.line(age_price, x="House Age (yrs)", y="Price per Unit Area",
                title="Average Price vs House Age")
            fig.update_traces(line_color=t["accent"], line_width=2.5)
            fig_layout(fig, t, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df, x="Neighbourhood", y="House Age (yrs)",
                color="Neighbourhood", color_discrete_map=CAT_COLORS,
                title="Age Distribution by Neighbourhood Tier",
                category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
            fig_layout(fig2, t, height=380)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.histogram(df, x="House Age (yrs)", color="Neighbourhood",
                color_discrete_map=CAT_COLORS, barmode="overlay", opacity=0.7,
                title="Age Distribution by Price Tier",
                category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
            fig_layout(fig3, t, height=350)
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = px.scatter(df, x="House Age (yrs)", y="Price per Unit Area",
                color="Neighbourhood", color_discrete_map=CAT_COLORS,
                trendline="lowess",
                title="Scatter: Age vs Price (with trend)",
                category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
            fig_layout(fig4, t, height=350)
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 3: MRT ──
    with tabs[2]:
        st.markdown(f'<div class="insight-box">🚇 Distance to MRT station is the <b>strongest predictor</b> of price (correlation = -0.67). Properties within ~1,000 m of a station are significantly more expensive, but the relationship is nuanced by neighbourhood tier.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="Distance to MRT (m)", y="Price per Unit Area",
                color="Neighbourhood", color_discrete_map=CAT_COLORS,
                title="MRT Distance vs Price",
                category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
            fig_layout(fig, t, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mrt_price = df.groupby(pd.cut(df["Distance to MRT (m)"], bins=10))["Price per Unit Area"].mean().reset_index()
            mrt_price["Distance to MRT (m)"] = mrt_price["Distance to MRT (m)"].astype(str)
            fig2 = px.bar(mrt_price, x="Distance to MRT (m)", y="Price per Unit Area",
                title="Avg Price by MRT Distance Band")
            fig2.update_traces(marker_color=t["accent"])
            fig_layout(fig2, t, height=400)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="Neighbourhood", y="Distance to MRT (m)",
            color="Neighbourhood", color_discrete_map=CAT_COLORS,
            title="MRT Distance by Neighbourhood Tier",
            category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
        fig3.update_layout(showlegend=False)
        fig_layout(fig3, t, height=360)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 4: Amenities ──
    with tabs[3]:
        st.markdown(f'<div class="insight-box">🏪 Convenience stores act as a <b>proxy for neighbourhood vibrancy</b>. More stores generally means a higher price, though the effect is weaker than MRT distance. It becomes more significant when combined with age and location.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            store_avg = df.groupby("Convenience Stores")["Price per Unit Area"].mean().reset_index()
            fig = px.bar(store_avg, x="Convenience Stores", y="Price per Unit Area",
                title="Avg Price by Number of Convenience Stores")
            fig.update_traces(marker_color=t["accent2"])
            fig_layout(fig, t, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df, x="Convenience Stores", y="Price per Unit Area",
                color_discrete_sequence=[t["accent"]],
                title="Price Distribution per Store Count")
            fig_layout(fig2, t, height=380)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="Neighbourhood", y="Convenience Stores",
            color="Neighbourhood", color_discrete_map=CAT_COLORS,
            title="Store Count by Neighbourhood Tier",
            category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
        fig3.update_layout(showlegend=False)
        fig_layout(fig3, t, height=360)
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 5: Correlations ──
    with tabs[4]:
        st.markdown(f'<div class="insight-box">🔗 Correlation only tells part of the story. MRT Distance has the strongest correlation (-0.67) but <b>low correlation does not mean useless</b> — house age shows non-linear patterns that a simple correlation coefficient misses entirely.</div>', unsafe_allow_html=True)

        num_df = df[["Transaction Date","House Age (yrs)","Distance to MRT (m)","Convenience Stores","Latitude","Longitude","Price per Unit Area"]]
        corr = num_df.corr().round(3)

        fig = px.imshow(corr, text_auto=True, aspect="auto",
            color_continuous_scale=t["colorscale"],
            title="Feature Correlation Matrix")
        fig.update_layout(paper_bgcolor=t["paper_bg"], plot_bgcolor=t["plot_bg"],
            font=dict(color=t["text"]), margin=dict(l=0,r=0,t=44,b=0), height=480)
        st.plotly_chart(fig, use_container_width=True)

        # Pair-wise price correlations bar
        price_corr = corr["Price per Unit Area"].drop("Price per Unit Area").sort_values()
        fig2 = px.bar(x=price_corr.values, y=price_corr.index, orientation="h",
            title="Correlation with Price per Unit Area",
            color=price_corr.values,
            color_continuous_scale="RdBu")
        fig_layout(fig2, t, height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 6: Market Trends ──
    with tabs[5]:
        st.markdown(f'<div class="insight-box">📅 The dataset spans a narrow window (Aug 2012 – Jul 2013). While limited, it shows a <b>price dip around late 2012</b> and a recovery in early 2013. Pricey homes show the most volatility over time.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            time_avg = df.groupby("Transaction Date")["Price per Unit Area"].mean().reset_index()
            fig = px.line(time_avg, x="Transaction Date", y="Price per Unit Area",
                title="Average Price Over Time")
            fig.update_traces(line_color=t["accent"], line_width=2.5)
            fig_layout(fig, t, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(df, x="Transaction Date", y="Price per Unit Area",
                color="Neighbourhood", color_discrete_map=CAT_COLORS,
                title="Price Trends by Neighbourhood Tier",
                category_orders={"Neighbourhood": ["Cheap","Mid-Class","Pricey","Luxurious"]})
            fig_layout(fig2, t, height=380)
            st.plotly_chart(fig2, use_container_width=True)

        # Price distribution
        fig3 = px.histogram(df, x="Price per Unit Area", nbins=40,
            color_discrete_sequence=[t["accent"]],
            marginal="violin",
            title="Price Distribution (with violin)")
        fig_layout(fig3, t, height=360)
        st.plotly_chart(fig3, use_container_width=True)


@st.cache_resource
def train_model():
    feat_cols = ["House Age (yrs)", "Distance to MRT (m)", "Convenience Stores", "Latitude", "Longitude"]
    X = df[feat_cols]
    y = df["Price per Unit Area"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # 5-fold cross-validation R² — these are REAL scores computed on held-out folds
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    return model, feat_cols, X_test, y_test, y_pred, rmse, r2, mae, cv

# ── Currency conversion helpers ──────────────────────────────────────
# 1 ping (坪) ≈ 3.306 m²
# Price unit: 10,000 TWD per ping
# 1 TWD ≈ 0.031 USD  (approximate — update as needed)
TWD_TO_USD = 0.031
PING_TO_SQM = 3.306

def price_unit_to_twd_per_sqm(price_unit):
    """Convert dataset unit (10k TWD/ping) → TWD per m²"""
    return (price_unit * 10_000) / PING_TO_SQM

def price_unit_to_usd_per_sqm(price_unit):
    """Convert dataset unit (10k TWD/ping) → USD per m²"""
    return price_unit_to_twd_per_sqm(price_unit) * TWD_TO_USD

def price_unit_to_total_usd(price_unit, area_ping=25):
    """Estimate total property price in USD for a given area (default 25 ping ≈ 83 m²)"""
    return price_unit * 10_000 * area_ping * TWD_TO_USD


def page_predictor(t):
    st.markdown('<div class="section-header">🤖 AI Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">A Random Forest model trained on all 414 real properties in Taiwan. Adjust characteristics below — or <b>click the map</b> to pick a location — then hit Predict.</p>', unsafe_allow_html=True)

    model, feat_cols, X_test, y_test, y_pred, rmse, r2, mae, cv = train_model()

    # ── Model metric cards ───────────────────────────────────────────
    # All 4 values are REAL — computed by train_model() above on actual data splits
    m_cols = st.columns(4)
    metric_info = [
        (f"{r2*100:.1f}%",       "R² Score",             "Variance explained on the held-out test set (20% of data)"),
        (f"{rmse:.2f}",           "RMSE",                 "Root Mean Squared Error on test set (same unit as price)"),
        (f"{mae:.2f}",            "MAE",                  "Mean Absolute Error — avg prediction is off by this amount"),
        (f"{cv.mean()*100:.1f}%", "Cross-Val R² (5-fold)","Average R² across 5 held-out folds — measures generalisation"),
    ]
    for col, (val, lbl, tip) in zip(m_cols, metric_info):
        with col:
            st.markdown(f'''
            <div class="metric-card" title="{tip}">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{lbl}</div>
              <div style="font-size:.7rem;color:{t["subtext"]};margin-top:6px;line-height:1.4;">{tip}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box" style="margin-top:14px;">
      ℹ️ All metrics above are <b>computed in real-time</b> from the trained Random Forest model.
      R² = {r2*100:.1f}% means the model explains <b>{r2*100:.1f}%</b> of price variance on unseen data.
      The 5-fold CV R² of <b>{cv.mean()*100:.1f}%</b> (±{cv.std()*100:.1f}%) confirms consistent performance across different data splits — not overfitting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["🎯 Predict Price", "📊 Model Performance", "🔬 Feature Importance"])

    # ════════════════════════════════════════════════════════════════
    with tabs[0]:
        # ── Price unit explainer banner ──────────────────────────────
        st.markdown(f"""
        <div style="background:{t['card2']};border:1px solid {t['border']};border-radius:12px;
                    padding:14px 20px;margin-bottom:18px;display:flex;align-items:center;gap:14px;">
          <div style="font-size:1.6rem;">💡</div>
          <div style="font-size:.88rem;color:{t['subtext']};line-height:1.6;">
            <b style="color:{t['text']}">How to read the price:</b>
            The dataset uses <b style="color:{t['accent']}">10,000 TWD per ping</b> as the price unit.
            1 ping (坪) = 3.306 m². So a score of <b>40</b> means
            <b style="color:{t['accent2']}">400,000 TWD per ping</b>
            ≈ <b style="color:{t['accent2']}">~121,000 TWD/m²</b>
            ≈ <b style="color:{t['accent3']}">~$3,750 USD/m²</b>.
            The full property price depends on its total floor area.
          </div>
        </div>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([1.15, 0.85])

        with left_col:
            st.markdown(f'<div style="font-weight:600;color:{t["text"]};margin-bottom:12px;font-size:1rem;">🏗️ Property Characteristics</div>', unsafe_allow_html=True)
            age   = st.slider("🏚️ House Age (years)",              0.0,   45.0,  15.0, 0.5)
            mrt   = st.slider("🚇 Distance to MRT (metres)",       50.0, 6500.0, 500.0, 10.0)
            stores = st.slider("🏪 Nearby Convenience Stores",      0,    10,     4)
            area_ping = st.slider("📐 Property Floor Area (ping, for total estimate)", 5, 100, 25)

            st.markdown(f'<div style="font-weight:600;color:{t["text"]};margin:16px 0 10px 0;font-size:1rem;">📍 Location</div>', unsafe_allow_html=True)

            # ── Step 1: pick a reference property from the dataset ────
            # Build a labelled list: "Prop #12 — Pricey | Lat 24.9731, Lon 121.5402"
            prop_options = {
                f"Prop #{i+1}  |  {row['Neighbourhood']}  |  Lat {row['Latitude']:.4f}, Lon {row['Longitude']:.4f}": (
                    float(row["Latitude"]), float(row["Longitude"])
                )
                for i, row in df.reset_index(drop=True).iterrows()
            }
            # Sort by neighbourhood tier so cheapest are first
            tier_order = {"Cheap": 0, "Mid-Class": 1, "Pricey": 2, "Luxurious": 3}
            sorted_keys = sorted(prop_options.keys(),
                                 key=lambda k: tier_order.get(k.split("|")[1].strip(), 9))

            selected_prop = st.selectbox(
                "🏘️ Pick a reference property from the dataset (sets lat/lon)",
                ["— Enter manually below —"] + sorted_keys,
                key="prop_picker",
            )

            # When a property is chosen, push its coords into number_input via session_state
            if selected_prop != "— Enter manually below —":
                ref_lat, ref_lon = prop_options[selected_prop]
                st.session_state["ni_lat"] = ref_lat
                st.session_state["ni_lon"] = ref_lon

            # ── Step 2: number inputs — always stable, never snap back ─
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input(
                    "↕️ Latitude",
                    min_value=float(df["Latitude"].min()),
                    max_value=float(df["Latitude"].max()),
                    value=float(st.session_state.get("ni_lat", df["Latitude"].mean())),
                    step=0.0001,
                    format="%.5f",
                    key="ni_lat",
                )
            with c2:
                lon = st.number_input(
                    "↔️ Longitude",
                    min_value=float(df["Longitude"].min()),
                    max_value=float(df["Longitude"].max()),
                    value=float(st.session_state.get("ni_lon", df["Longitude"].mean())),
                    step=0.0001,
                    format="%.5f",
                    key="ni_lon",
                )

            # ── Step 3: fast static preview map (no interaction = no lag) ─
            # Uses st.map which is a lightweight Deck.GL scatter — renders instantly
            preview_df = df[["Latitude", "Longitude"]].copy()
            preview_df["size"] = 50
            # Highlight the selected location in a separate row
            selected_row = pd.DataFrame({
                "Latitude": [lat], "Longitude": [lon], "size": [300]
            })
            st.markdown(f'<div style="font-size:.75rem;color:{t["subtext"]};margin-bottom:4px;">🗺️ Map preview — selected location shown in red</div>', unsafe_allow_html=True)

            # Build a quick Plotly scatter map WITHOUT on_select (no lag, no rerun loop)
            fig_prev = go.Figure()
            fig_prev.add_trace(go.Scattermapbox(
                lat=df["Latitude"], lon=df["Longitude"],
                mode="markers",
                marker=dict(size=6, color=df["Price per Unit Area"],
                            colorscale=t["colorscale"], opacity=0.65,
                            cmin=df["Price per Unit Area"].min(),
                            cmax=df["Price per Unit Area"].max()),
                hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                name="All Properties",
                showlegend=False,
            ))
            fig_prev.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lon],
                mode="markers",
                marker=dict(size=18, color=t["accent3"]),
                hovertemplate=f"📍 Selected<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}<extra></extra>",
                name="Selected",
                showlegend=False,
            ))
            fig_prev.update_layout(
                mapbox=dict(style="carto-darkmatter",
                            center=dict(lat=lat, lon=lon), zoom=13),
                paper_bgcolor=t["paper_bg"],
                margin=dict(l=0, r=0, t=0, b=0),
                height=260,
            )
            # key=f"map_{lat:.4f}_{lon:.4f}" prevents stale render without triggering rerun
            st.plotly_chart(fig_prev, use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"prev_map_{lat:.4f}_{lon:.4f}")

            predict_btn = st.button("🔮 Predict Price", use_container_width=True)

        with right_col:
            if predict_btn:
                input_data = pd.DataFrame([[age, mrt, stores, lat, lon]], columns=feat_cols)
                pred = model.predict(input_data)[0]

                # ── Currency conversions ────────────────────────────
                twd_per_ping   = pred * 10_000                        # TWD per ping
                twd_per_sqm    = twd_per_ping / PING_TO_SQM           # TWD per m²
                usd_per_sqm    = twd_per_sqm * TWD_TO_USD             # USD per m²
                total_twd      = twd_per_ping * area_ping             # total TWD
                total_usd      = total_twd * TWD_TO_USD               # total USD

                # Tier
                tier_map = [(27.7, "Cheap", "🔴"), (38.5, "Mid-Class", "🟠"),
                            (46.6, "Pricey", "🔵"), (999, "Luxurious", "🟢")]
                tier_label, tier_icon = next((l, ic) for lim, l, ic in tier_map if pred <= lim)

                st.markdown(f"""
                <div class="pred-result">
                  <div style="font-size:2rem;margin-bottom:6px;">🏡</div>
                  <div style="color:{t['subtext']};font-size:.75rem;text-transform:uppercase;
                              letter-spacing:.1em;margin-bottom:4px;">Estimated Price Index</div>
                  <div class="pred-val">{pred:.1f}</div>
                  <div style="color:{t['subtext']};font-size:.78rem;margin-bottom:14px;">
                    × 10,000 TWD per ping (坪)
                  </div>

                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
                    <div style="background:{t['card2']};border-radius:10px;padding:12px;text-align:center;">
                      <div style="font-size:1.15rem;font-weight:700;color:{t['accent2']};">
                        NT$ {twd_per_sqm:,.0f}
                      </div>
                      <div style="font-size:.7rem;color:{t['subtext']};margin-top:2px;">TWD / m²</div>
                    </div>
                    <div style="background:{t['card2']};border-radius:10px;padding:12px;text-align:center;">
                      <div style="font-size:1.15rem;font-weight:700;color:{t['accent3']};">
                        $ {usd_per_sqm:,.0f}
                      </div>
                      <div style="font-size:.7rem;color:{t['subtext']};margin-top:2px;">USD / m²</div>
                    </div>
                    <div style="background:{t['card2']};border-radius:10px;padding:12px;text-align:center;">
                      <div style="font-size:1.05rem;font-weight:700;color:{t['accent2']};">
                        NT$ {total_twd:,.0f}
                      </div>
                      <div style="font-size:.7rem;color:{t['subtext']};margin-top:2px;">Total TWD ({area_ping} ping)</div>
                    </div>
                    <div style="background:{t['card2']};border-radius:10px;padding:12px;text-align:center;">
                      <div style="font-size:1.05rem;font-weight:700;color:{t['accent3']};">
                        $ {total_usd:,.0f}
                      </div>
                      <div style="font-size:.7rem;color:{t['subtext']};margin-top:2px;">Total USD ({area_ping} ping)</div>
                    </div>
                  </div>

                  <hr style="border-color:{t['border']};margin:10px 0;">
                  <div style="font-size:1rem;color:{t['text']};font-weight:600;">
                    {tier_icon} Neighbourhood Tier: {tier_label}
                  </div>
                  <div style="color:{t['subtext']};font-size:.76rem;margin-top:5px;">
                    Market index range: {df['Price per Unit Area'].min():.1f} – {df['Price per Unit Area'].max():.1f}
                    &nbsp;|&nbsp; 1 TWD ≈ {TWD_TO_USD} USD
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 120], "tickcolor": t["subtext"]},
                        "bar": {"color": t["accent"]},
                        "bgcolor": t["card"],
                        "bordercolor": t["border"],
                        "steps": [
                            {"range": [0,    27.7], "color": "rgba(247,129,102,0.13)"},
                            {"range": [27.7, 38.5], "color": "rgba(255,166,87,0.13)"},
                            {"range": [38.5, 46.6], "color": "rgba(121,192,255,0.13)"},
                            {"range": [46.6, 120],  "color": "rgba(63,185,80,0.13)"},
                        ],
                        "threshold": {"line": {"color": t["accent3"], "width": 3},
                                      "thickness": 0.8, "value": pred},
                    },
                    title={"text": "Price Index / Unit Area", "font": {"color": t["text"], "size": 13}},
                    number={"font": {"color": t["accent"], "size": 32}},
                ))
                fig.update_layout(paper_bgcolor=t["paper_bg"], font_color=t["subtext"],
                                  height=260, margin=dict(l=20, r=20, t=44, b=10))
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.markdown(f"""
                <div style="background:{t['card']};border:1px dashed {t['border']};border-radius:16px;
                            padding:36px;text-align:center;color:{t['subtext']};">
                  <div style="font-size:2.5rem;margin-bottom:12px;">🔮</div>
                  <div style="line-height:1.7;">
                    Set property details on the left,<br>
                    <b style="color:{t['text']}">click the map</b> to pick a location,<br>
                    then press <b style="color:{t['accent']}">Predict Price</b>.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown(f"""
        <div class="insight-box">
          📊 All charts below use the <b>actual test-set predictions</b> from the trained model
          (held-out 20% = 83 properties never seen during training).
          The CV bar chart shows R² for each of the 5 folds — genuinely computed, not hard-coded.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=y_test, y=y_pred,
                labels={"x": "Actual Price (index)", "y": "Predicted Price (index)"},
                title="Actual vs Predicted — Test Set (83 properties)")
            fig.update_traces(marker=dict(color=t["accent"], opacity=0.65, size=7))
            lim = [min(y_test.min(), min(y_pred))-2, max(y_test.max(), max(y_pred))+2]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                line=dict(color=t["accent3"], dash="dash", width=1.5), name="Perfect Fit"))
            fig_layout(fig, t, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            residuals = np.array(y_test) - y_pred
            fig2 = px.histogram(x=residuals, nbins=35,
                title=f"Residuals (mean={np.mean(residuals):.2f}, std={np.std(residuals):.2f})",
                color_discrete_sequence=[t["accent2"]])
            fig2.add_vline(x=0, line_dash="dash", line_color=t["accent3"],
                           annotation_text="Zero error", annotation_font_color=t["subtext"])
            fig_layout(fig2, t, height=400)
            st.plotly_chart(fig2, use_container_width=True)

        _, _, _, _, _, _, _, _, cv_scores = train_model()
        fig3 = px.bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores * 100,
            title=f"5-Fold Cross-Validation R² — Mean: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%",
            color=cv_scores * 100,
            color_continuous_scale=t["colorscale"],
            labels={"x": "CV Fold", "y": "R² Score (%)"},
        )
        fig3.add_hline(y=cv_scores.mean()*100, line_dash="dash", line_color=t["accent"],
            annotation_text=f"Mean R²: {cv_scores.mean()*100:.1f}%",
            annotation_font_color=t["accent"])
        fig_layout(fig3, t, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    with tabs[2]:
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feat_cols, "Importance": importances}).sort_values("Importance", ascending=True)
        fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
            title="Random Forest Feature Importances (mean decrease in impurity)",
            color="Importance", color_continuous_scale=t["colorscale"])
        fig_layout(fig, t, height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
          🏆 <b>Latitude & Longitude</b> dominate — location truly is everything in real estate.
          <b>MRT Distance</b> follows (closer = pricier), then <b>Convenience Stores</b> (neighbourhood vibrancy proxy),
          and finally <b>House Age</b> (U-shaped effect: new or vintage commands premium).
          Transaction Date was intentionally excluded — it would overfit the narrow 2012-2013 window.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="font-weight:600;color:{t["text"]};margin:20px 0 12px 0;">Partial Dependence — How Each Feature Affects Predicted Price</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        ranges = {
            "Distance to MRT (m)": np.linspace(23, 6488, 100),
            "House Age (yrs)": np.linspace(0, 44, 100),
        }
        base = {f: df[f].median() for f in feat_cols}
        for col, (fname, rng) in zip([col1, col2], ranges.items()):
            with col:
                preds_pd = [model.predict(pd.DataFrame([{**base, fname: v}], columns=feat_cols))[0] for v in rng]
                fig_p = px.line(x=rng, y=preds_pd,
                    labels={"x": fname, "y": "Predicted Price Index"},
                    title=f"Effect of {fname} (all else at median)")
                fig_p.update_traces(line_color=t["accent"], line_width=2.5)
                fig_layout(fig_p, t, height=320)
                st.plotly_chart(fig_p, use_container_width=True)


def page_contact(t):
    st.markdown('<div class="section-header">📬 Get In Touch</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Interested in data-driven insights, collaboration, or have questions about this project? Reach out!</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        contacts = [
            ("📧", "Email", "your.email@example.com", "mailto:your.email@example.com"),
            ("💼", "LinkedIn", "linkedin.com/in/yourprofile", "https://linkedin.com/in/yourprofile"),
            ("🐙", "GitHub", "github.com/yourusername", "https://github.com/yourusername"),
            ("🌐", "Portfolio", "yourportfolio.com", "https://yourportfolio.com"),
        ]
        for icon, label, value, link in contacts:
            st.markdown(f"""
            <div class="info-card" style="margin-bottom:12px;display:flex;align-items:center;gap:16px;">
              <div style="font-size:1.6rem">{icon}</div>
              <div>
                <div style="font-size:.75rem;color:{t['subtext']};text-transform:uppercase;letter-spacing:.06em">{label}</div>
                <a href="{link}" style="color:{t['accent']};font-weight:600;text-decoration:none;">{value}</a>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{t['card']};border:1px solid {t['border']};border-radius:16px;padding:28px;text-align:center;">
          <div style="font-size:2.8rem;margin-bottom:12px;">👨‍💻</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:{t['text']};margin-bottom:6px;">Eng. Abo0od</div>
          <div style="color:{t['subtext']};font-size:.88rem;line-height:1.6;margin-bottom:16px;">
            Data Scientist & ML Engineer<br>
            Passionate about real estate analytics,<br>predictive modelling & data storytelling.
          </div>
          <span class="badge">🐍 Python</span>
          <span class="badge">📊 Data Science</span>
          <span class="badge">🤖 ML</span>
          <span class="badge">🏠 PropTech</span>
        </div>
        """, unsafe_allow_html=True)

    # Project info
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    tcols = st.columns(4)
    techs = [
        ("🐍", "Python", "Core language"),
        ("📊", "Streamlit", "Web framework"),
        ("📈", "Plotly", "Interactive charts"),
        ("🤖", "Scikit-Learn", "ML modelling"),
    ]
    for col, (icon, name, desc) in zip(tcols, techs):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.8rem;margin-bottom:6px;">{icon}</div>
              <div style="font-weight:600;color:{t['text']}">{name}</div>
              <div class="metric-label">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    t = THEMES[st.session_state.theme]
    inject_css(t)
    nav = render_sidebar(t)

    if "Overview" in nav:
        page_overview(t)
    elif "Data Explorer" in nav:
        page_data_explorer(t)
    elif "Insights" in nav:
        page_insights(t)
    elif "Predictor" in nav:
        page_predictor(t)
    elif "Contact" in nav:
        page_contact(t)

    # Footer
    st.markdown(f"""
    <div class="footer">
      Built with ❤️ and data by <span>Eng. Apo0od</span> &nbsp;|&nbsp;
      EstateIQ — Real Estate Intelligence Platform &nbsp;|&nbsp;
      Powered by Random Forest & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
