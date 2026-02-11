# ntu_dsai_module_1_capstone_streamlit_app_final

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
    ''',
    unsafe_allow_html=True
)

st.title("Rebalancing the Singapore Job Market for Economic Sustainability")
st.subheader("Signs of Structural Misalignments")

# ======================================================
# DATA ENGINE: CACHED PRE-PROCESSING & FIXED CATEGORIES
# ======================================================
@st.cache_data
def load_and_prepare_data():
    path = "SGJobData_cleaned_processed_compressed.csv.gz"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # 1. Date Standardization
    df['metadata_originalPostingDate'] = pd.to_datetime(df['metadata_originalPostingDate'], errors='coerce')
    df = df.dropna(subset=['metadata_originalPostingDate'])
    df['month_year'] = df['metadata_originalPostingDate'].dt.to_period('M').dt.to_timestamp()
    
    # 2. Vectorized Math & Salary Logic
    if 'salary_minimum' in df.columns and 'salary_maximum' in df.columns:
        df['average_salary'] = (df['salary_minimum'] + df['salary_maximum']) / 2

    # 3. Fast Category Parsing
    def parse_categories(cat):
        if pd.isna(cat): return []
        try:
            parsed = ast.literal_eval(cat)
            return parsed if isinstance(parsed, list) else []
        except: return []

    df['primary_category'] = df['categories'].apply(parse_categories).apply(
        lambda x: x[0]['category'] if len(x) > 0 and 'category' in x[0] else "Unknown"
    )

    # 4. PRESERVE FIXED STORY
    avg_apps_global = df.groupby('primary_category', observed=True)['metadata_totalNumberJobApplication'].mean().sort_values()
    bottom10_cats = avg_apps_global.head(10).index.tolist()
    top10_cats = avg_apps_global.tail(10).index.tolist()
    
    # 5. Summary table for Dynamic Sections
    trend_summary = df.groupby(['month_year', 'primary_category', 'employmentTypes'], observed=True).agg(
        job_count=('month_year', 'count'),
        app_sum=('metadata_totalNumberJobApplication', 'sum'),
        vac_sum=('numberOfVacancies', 'sum')
    ).reset_index()

    return df, trend_summary, bottom10_cats, top10_cats

df_full, df_trend, b10_cats, t10_cats = load_and_prepare_data()

# ======================================================
# GLOBAL DASHBOARD FILTERS (Top Panel)
# ======================================================
st.write("### 🛠️ Visual Explorer")
c_f1, c_f2, c_f3 = st.columns([1, 1, 2])

with c_f1:
    category_filter = st.multiselect("Job Category", sorted(df_trend['primary_category'].unique()))
with c_f2:
    emp_type_filter = st.multiselect("Employment Type", sorted(df_trend['employmentTypes'].dropna().unique()))
with c_f3:
    min_date = df_trend['month_year'].min().to_pydatetime()
    max_date = df_trend['month_year'].max().to_pydatetime()
    date_range = st.slider("Analysis Period", min_date, max_date, (min_date, max_date), format="MM/YYYY")

# --- Filtering Logic ---
start_ts, end_ts = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
filtered_trend = df_trend.copy()
if category_filter:
    filtered_trend = filtered_trend[filtered_trend['primary_category'].isin(category_filter)]
if emp_type_filter:
    filtered_trend = filtered_trend[filtered_trend['employmentTypes'].isin(emp_type_filter)]
filtered_trend = filtered_trend[(filtered_trend['month_year'] >= start_ts) & (filtered_trend['month_year'] <= end_ts)]

# Filter full dataset for advanced metrics
df_dynamic_full = df_full.copy()
if category_filter:
    df_dynamic_full = df_dynamic_full[df_dynamic_full['primary_category'].isin(category_filter)]
if emp_type_filter:
    df_dynamic_full = df_dynamic_full[df_dynamic_full['employmentTypes'].isin(emp_type_filter)]
df_dynamic_full = df_dynamic_full[(df_dynamic_full['metadata_originalPostingDate'] >= start_ts) & 
                                  (df_dynamic_full['metadata_originalPostingDate'] <= end_ts)]

# ======================================================
# SECTION 1: DYNAMIC KPI DASHBOARD (3x2 Matrix)
# ======================================================
st.markdown("---")

# Dynamic Statistics Calculations
total_jobs = filtered_trend['job_count'].sum()
total_apps = filtered_trend['app_sum'].sum()
total_vacs = filtered_trend['vac_sum'].sum()
total_hirers = df_dynamic_full['postedCompany_name'].nunique()
total_views = df_dynamic_full['metadata_totalNumberOfView'].sum()

# NEW/TWEAKED CALCULATIONS
avg_apps_dynamic = total_apps / total_jobs if total_jobs > 0 else 0
avg_views_dynamic = total_views / total_jobs if total_jobs > 0 else 0

col_metrics, col_wordcloud = st.columns([2.2, 3.3])

with col_metrics:
    # Added vertical spacer to push matrix down and sync with wordcloud height
    st.markdown("<div style='margin-top: 45px;'></div>", unsafe_allow_html=True)
    
    # Row 1: Jobs & Applications
    r1_k1, r1_k2 = st.columns(2)
    r1_k1.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Job Postings</div><div class='kpi-value'>{total_jobs:,}</div></div>", unsafe_allow_html=True)
    r1_k2.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Applications</div><div class='kpi-value'>{int(total_apps):,}</div></div>", unsafe_allow_html=True)
    
    # Row 2: Vacancies & Avg Views / Job
    r2_k1, r2_k2 = st.columns(2)
    r2_k1.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Vacancies</div><div class='kpi-value'>{int(total_vacs):,}</div></div>", unsafe_allow_html=True)
    r2_k2.markdown(f"<div class='kpi-box'><div class='kpi-title'>Avg Views / Job</div><div class='kpi-value'>{avg_views_dynamic:.1f}</div></div>", unsafe_allow_html=True)

    # Row 3: Total Hirers & Avg Applicants / Job
    r3_k1, r3_k2 = st.columns(2)
    r3_k1.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Hirers</div><div class='kpi-value'>{total_hirers:,}</div></div>", unsafe_allow_html=True)
    r3_k2.markdown(f"<div class='kpi-box'><div class='kpi-title'>Avg Applicants / Job</div><div class='kpi-value'>{avg_apps_dynamic:.1f}</div></div>", unsafe_allow_html=True)

with col_wordcloud:
    if not df_dynamic_full.empty:
        text = " ".join(df_dynamic_full['title'].fillna('').astype(str))
        if text.strip():
            wordcloud = WordCloud(
                background_color="#0e1117", colormap='Blues', 
                width=1000, height=640, max_words=80
            ).generate(text)
            fig_wc, ax = plt.subplots(figsize=(10, 6.4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            fig_wc.patch.set_facecolor('#0e1117')
            st.markdown("<div style='text-align: center; color: #8b949e; font-size: 14px; margin-bottom: 5px; font-weight: bold;'>Key Trending Job Postings</div>", unsafe_allow_html=True)
            st.pyplot(fig_wc)
        else:
            st.info("No titles found.")
    else:
        st.info("No data for WordCloud.")

# ======================================================
# SECTION 2: DYNAMIC TIME SERIES TREND
# ======================================================
st.subheader("Job Market Dynamics Over Time")

ts = filtered_trend.groupby('month_year').agg(
    job_postings=('job_count', 'sum'),
    applicants=('app_sum', 'sum'),
    vacancies=('vac_sum', 'sum')
).reset_index()

fig_ts = go.Figure()
fig_ts.add_scatter(x=ts['month_year'], y=ts['job_postings'], name="Job Postings", line=dict(color='maroon', width=3))
fig_ts.add_scatter(x=ts['month_year'], y=ts['applicants'], name="Applicants", yaxis="y2", line=dict(color='lime'))
fig_ts.add_scatter(x=ts['month_year'], y=ts['vacancies'], name="Vacancies", yaxis="y2", line=dict(color='goldenrod'))

fig_ts.update_layout(
    template="plotly_dark", height=450,
    yaxis=dict(title="Postings / Vacancies", showgrid=False),
    yaxis2=dict(title="Applicants", overlaying='y', side='right', showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_ts, width="stretch")

# ======================================================
# SECTION 3: FIXED DEEP DIVE - THE MARKET STORY
# ======================================================
st.markdown("---")
st.subheader("Deep Dive: Contrasting Landscapes of Laggards & Leaders")

# 1. Global Comparison of Average Apps
global_avg_stats = df_full.groupby('primary_category', observed=True)['metadata_totalNumberJobApplication'].mean().round(2)
fig_comp = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Bottom 10 Avg Apps", "Top 10 Avg Apps"))
fig_comp.add_trace(go.Bar(x=b10_cats, y=global_avg_stats.loc[b10_cats], marker_color='maroon'), row=1, col=1)
fig_comp.add_trace(go.Bar(x=t10_cats, y=global_avg_stats.loc[t10_cats], marker_color='lime'), row=1, col=2)
fig_comp.update_layout(template="plotly_dark", height=450, showlegend=False, title_text="Comparison of Average Applicants per Posting")
st.plotly_chart(fig_comp, width="stretch")

# 2. Ratio Chart (Apps to Vacancies)
df_full['apps_per_vacancy'] = df_full['metadata_totalNumberJobApplication'] / df_full['numberOfVacancies']
global_ratio_stats = df_full.groupby('primary_category', observed=True)['apps_per_vacancy'].mean().round(2)
fig_ratio = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Bottom 10 Ratio", "Top 10 Ratio"))
fig_ratio.add_trace(go.Bar(x=b10_cats, y=global_ratio_stats.loc[b10_cats], marker_color='maroon'), row=1, col=1)
fig_ratio.add_trace(go.Bar(x=t10_cats, y=global_ratio_stats.loc[t10_cats], marker_color='lime'), row=1, col=2)
fig_ratio.update_layout(template="plotly_dark", height=450, showlegend=False, title_text="Applications per Vacancy Ratio")
st.plotly_chart(fig_ratio, width="stretch")

# 3. Salary Box Plot
df_full['log_salary'] = np.log1p(df_full['average_salary'])
fig_salary = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Bottom 10 (Log Salary)", "Top 10 (Log Salary)"))
for cat in b10_cats:
    fig_salary.add_trace(go.Box(y=df_full[df_full.primary_category==cat]['log_salary'], name=cat, boxmean=True, boxpoints=False), row=1, col=1)
for cat in t10_cats:
    fig_salary.add_trace(go.Box(y=df_full[df_full.primary_category==cat]['log_salary'], name=cat, boxmean=True, boxpoints=False), row=1, col=2)
fig_salary.update_layout(template="plotly_dark", height=600, showlegend=False, title_text="Comparison of Log-Transformed Salary Distributions")
st.plotly_chart(fig_salary, width="stretch")

# ======================================================
# SECTION 4: CORRELATION HEATMAP
# ======================================================
st.markdown("---")
st.subheader("The Pay-Applicant Nexus: Weak Influence of Payscales on Applicant Volumes")
corr_cols = ['metadata_totalNumberJobApplication', 'average_salary', 'salary_minimum', 'salary_maximum']
existing = [c for c in corr_cols if c in df_full.columns]
if len(existing) > 1:
    corr = df_full[existing].dropna().corr()
    mask = np.tril(np.ones_like(corr, dtype=bool))
    df_corr_diag = corr.where(mask)
    fig_corr = px.imshow(df_corr_diag, text_auto=".2f", color_continuous_scale='RdBu_r', template='plotly_dark')
    st.plotly_chart(fig_corr, width="stretch")

# ======================================================
# INSIGHTS
# ======================================================
st.subheader("The Key Takeaways")
st.markdown(
    f'''
    * **Structural Misalignments**: Higher application counts don't always correlate with pay scales, suggesting skill gaps or job perception issues.
    * **Market Tension**: Certain sectors show a high "Applications per Vacancy" ratio, indicating oversaturation, while others remain critically underserved.
    * **Evidence of Imbalance**: The stark differences between the bottom versus top 10 job categories signal potential structural misalignments in the current Singapore workforce. 
    * **Long-term Sustainability**: This calls for targeted rebalancing to ensure Singapore's economic sustainability for the longer term.
    '''
)