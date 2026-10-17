"""Static, local spatial UI styling. No external assets or dynamic HTML inputs."""

import streamlit as st


def apply_spatial_style():
    st.html("""
    <style>
    .stApp {
      background: radial-gradient(ellipse at 15% 0%, #172b44 0%, #0b1424 52%);
    }
    [data-testid="stHeader"] { background: rgba(11, 20, 36, 0.94); }
    [data-testid="stSidebar"] {
      background: #101e32;
      border-right: 1px solid #344760;
    }
    [data-testid="stMainBlockContainer"] { max-width: 1320px; }
    h1 { letter-spacing: -0.035em; line-height: 1.15 !important; }
    h2, h3 { letter-spacing: -0.015em; }
    [data-testid="stMetric"] {
      background: #14243a;
      border: 1px solid #344760;
      border-radius: 16px;
      padding: 1rem 1.15rem;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
    /* WCAG 1.4.3: Streamlit fades captions to 60% opacity (4.0:1 on panels)
       and puts white text on the light primary blue (1.8:1). Navy on that
       blue is 10.4:1; full-opacity captions are at least 6.8:1. */
    [data-testid="stCaptionContainer"] { opacity: 1 !important; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #b8c8dc !important; }
    [data-testid^="stBaseButton-primary"],
    [data-testid^="stBaseButton-primary"] p,
    [data-testid="stMultiSelectTagsContainer"] [data-tag],
    [data-testid="stMultiSelectTagsContainer"] [data-tag] * { color: #0b1424 !important; }
    /* st.json numbers use an inline #cc6633 (3.8:1 on panels); this hue is 5.2:1+. */
    [data-testid="stJson"] .variable-value > [style*="rgb(204, 102, 51)"] {
      color: #f0995f !important;
    }
    [data-testid="stAlert"] p { color: #eef4fc; }
    button, input, textarea { font-variant-numeric: tabular-nums; }
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
      [data-testid="stMetric"] { padding: 0.8rem; }
      h1 { font-size: 2rem !important; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; }
    }
    </style>
    """)
