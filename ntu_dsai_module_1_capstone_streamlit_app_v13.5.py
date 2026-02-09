# ntu_dsai_module_1_capstone_streamlit_app_v13.5 (compressed csv gz file + app stabilisation)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG & THEME
# ======================================================
st.set_page_config(
    page_title="Singapore Job Market Analysis",
    layout="wide"
)

# Professional UI styling
st.markdown(
    '''
    <style>
    .kpi-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 0 8px rgba(0,229,255,0.2);
        border: 1px solid #30363d;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 20px; color: #8b949e; margin-bottom: 10px; }
    .kpi-value { font-size: 36px; font-weight: bold; color: #58a6ff; white-space: nowrap; }
    </style>
    ''', unsafe_allow_html=True
)

# ======================================================
# DATA LOADING (STABILIZED)
# ======================================================
@st.cache_data
def load_data():
    # Loading the compressed dataset
    df = pd.read_csv("SGJobData_cleaned_processed_compressed.csv.gz", compression='gzip')
    # CRITICAL: Ensure date columns are datetime objects for the slider to work
    df['metadata_originalPostingDate'] = pd.to_datetime(df['metadata_originalPostingDate'])
    return df

df_full = load_data()

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.header("Filter Dashboard")

# Period Slider Stability Fix
min_date = df_full['metadata_originalPostingDate'].min()
max_date = df_full['metadata_originalPostingDate'].max()

selected_range = st.sidebar.slider(
    "Select Posting Period",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
)

# Apply Date Filter
df = df_full[
    (df_full['metadata_originalPostingDate'] >= selected_range[0]) & 
    (df_full['metadata_originalPostingDate'] <= selected_range[1])
]

# Category Filter
all_cats = sorted(df_full['primary_category'].dropna().unique())
selected_cat = st.sidebar.multiselect("Job Categories", options=all_cats, default=None)
if selected_cat:
    df = df[df['primary_category'].isin(selected_cat)]

# ======================================================
# MAIN DASHBOARD
# ======================================================
st.title("🇸🇬 Singapore Job Market Insights")

# Check if dataframe is empty after filtering to prevent crashes
if df.empty:
    st.warning("No data available for the selected filters. Please adjust your criteria.")
    st.stop()

# KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Total Job Posts</div><div class="kpi-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with col2:
    avg_sal = df['average_salary'].mean()
    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Avg Salary (SGD)</div><div class="kpi-value">${avg_sal:,.0f}</div></div>', unsafe_allow_html=True)
with col3:
    total_apps = df['metadata_totalNumberJobApplication'].sum()
    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Total Applications</div><div class="kpi-value">{total_apps:,.0f}</div></div>', unsafe_allow_html=True)
with col4:
    market_comp = df['apps_per_vacancy'].mean()
    st.markdown(f'<div class="kpi-box"><div class="kpi-title">Apps per Vacancy</div><div class="kpi-value">{market_comp:.1f}</div></div>', unsafe_allow_html=True)

# THE PAY-APPLICANT NEXUS
st.markdown("---")
st.subheader("The Pay-Applicant Nexus: Weak Influence of Payscales on Applicant Volumes")
corr_cols = ['metadata_totalNumberJobApplication', 'average_salary', 'salary_minimum', 'salary_maximum']
existing = [c for c in corr_cols if c in df.columns]
if len(existing) > 1:
    corr = df[existing].dropna().corr()
    mask = np.tril(np.ones_like(corr, dtype=bool))
    df_corr_diag = corr.where(mask)
    fig_corr = px.imshow(df_corr_diag, text_auto=".2f", color_continuous_scale='RdBu_r', template='plotly_dark')
    # STABILITY FIX: Updated use_container_width to width='stretch'
    st.plotly_chart(fig_corr, width='stretch')

# INSIGHTS
st.subheader("The Key Takeaways")
st.markdown(
    f'''
    * **Structural Misalignments**: Higher application counts don't always correlate with pay scales[cite: 37].
    * **Market Tension**: Certain sectors show a high "Applications per Vacancy" ratio, indicating oversaturation[cite: 37].
    * **Evidence of Imbalance**: Stark differences between bottom and top 10 job categories signal potential workforce misalignments[cite: 37].
    * **Long-term Sustainability**: Calls for targeted rebalancing to ensure Singapore's economic sustainability[cite: 37].
    '''
)