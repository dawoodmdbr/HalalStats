import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  –  sidebar collapsed so hamburger ☰ is visible on load
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HalalStats — Hajj Analytics",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  –  minimal, flat, light, orange accents, zero gradients / glows
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --orange:   #D95F0A;
    --orange-2: #F4A96A;
    --orange-3: #FDF0E6;
    --black:    #111111;
    --grey-1:   #444444;
    --grey-2:   #888888;
    --grey-3:   #CCCCCC;
    --grey-4:   #F7F7F7;
    --white:    #FFFFFF;
    --border:   #E2E2E2;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: var(--white) !important;
    color: var(--black) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebarContent"] { padding: 28px 20px !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px; display: flex; flex-direction: column;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 3px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    color: var(--grey-1) !important;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--orange-3) !important;
    color: var(--orange) !important;
}

/* Main */
.block-container { padding: 32px 44px 64px 44px !important; max-width: 1180px !important; }

/* Page title */
.pg-title {
    font-size: 1.4rem; font-weight: 600; color: var(--black);
    border-bottom: 2px solid var(--orange);
    padding-bottom: 8px; margin-bottom: 6px; display: inline-block;
}
.pg-sub { font-size: 12.5px; color: var(--grey-2); margin-bottom: 28px; }

/* Section label */
.sec-label {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--grey-2); margin-bottom: 8px;
}

/* Stat strip */
.stat-strip {
    display: flex; gap: 0; margin-bottom: 32px;
    border: 1px solid var(--border); border-radius: 4px; overflow: hidden;
}
.stat-item {
    flex: 1; padding: 16px 20px;
    border-right: 1px solid var(--border); background: var(--white);
}
.stat-item:last-child { border-right: none; }
.stat-item .s-label {
    font-size: 10.5px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.6px; color: var(--grey-2); margin-bottom: 5px;
}
.stat-item .s-val {
    font-size: 1.5rem; font-weight: 600; color: var(--orange);
    font-family: 'IBM Plex Mono', monospace;
}
.stat-item .s-note { font-size: 10.5px; color: var(--grey-3); margin-top: 2px; }

/* Chart wrapper */
.chart-wrap {
    border: 1px solid var(--border); border-radius: 4px;
    padding: 18px 20px; margin-bottom: 18px; background: var(--white);
}
.chart-title { font-size: 12.5px; font-weight: 600; color: var(--black); margin-bottom: 12px; }

/* Divider */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* Tables */
.dataframe thead tr th {
    background: var(--grey-4) !important; color: var(--black) !important;
    font-weight: 600 !important; font-size: 11.5px !important;
}
.dataframe tbody tr:nth-child(even) { background: #FAFAFA !important; }
.dataframe { font-size: 12px !important; }

/* Buttons */
.stButton > button {
    background: var(--orange) !important; color: var(--white) !important;
    border: none !important; border-radius: 3px !important;
    font-weight: 500 !important; font-size: 13px !important;
    padding: 8px 22px !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* Selectbox */
[data-baseweb="select"] > div {
    border-color: var(--border) !important; border-radius: 3px !important;
    font-size: 13px !important;
}

/* Metrics */
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--orange) !important; font-family: 'IBM Plex Mono', monospace !important;
}

/* Alerts */
.stAlert { border-radius: 3px !important; font-size: 12.5px !important; }
[data-testid="stInfo"] {
    border-left: 3px solid var(--orange) !important;
    background: var(--orange-3) !important;
}

/* Expander */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 3px !important; }

/* Nav header */
.nav-header {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--grey-2); margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
ORANGE_SEQ = ["#D95F0A","#E8883A","#F4A96A","#F9C99A","#FDE0C5","#C04A05","#A03A04","#7A2C02"]

def T(fig, title=""):
    fig.update_layout(
        font_family="IBM Plex Sans",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title_text=title,
        title_font=dict(size=12.5, color="#111111", family="IBM Plex Sans"),
        title_x=0,
        margin=dict(l=48, r=20, t=44, b=44),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E2E2", borderwidth=1, font_size=11),
        xaxis=dict(gridcolor="#F2F2F2", linecolor="#E2E2E2", tickfont_size=11),
        yaxis=dict(gridcolor="#F2F2F2", linecolor="#E2E2E2", tickfont_size=11),
    )
    return fig

def chart(fig, title=""):
    st.markdown(f'<div class="chart-wrap"><div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.plotly_chart(T(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("synthetic_hajj_dataset.csv")
    df["Stay_Days"]           = df["Stay_Duration"].str.extract(r"(\d+)").astype(int)
    df["Spending_Per_Day"]    = (df["Estimated_Spending_SAR"] / df["Stay_Days"]).round(2)
    df["Spending_Per_Person"] = (df["Estimated_Spending_SAR"] / df["Group_Size"]).round(2)
    return df

df = load()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nav-header">🕌 HalalStats</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "Descriptive Statistics",
        "Confidence Intervals",
        "Probability & Distributions",
        "Regression & Predictions",
        "Raw Data",
    ])

    st.markdown("---")
    st.markdown('<div class="nav-header">Filters</div>', unsafe_allow_html=True)
    sel_country = st.multiselect("Country",   sorted(df["Country"].unique()),    default=sorted(df["Country"].unique()))
    sel_gender  = st.multiselect("Gender",    list(df["Gender"].unique()),        default=list(df["Gender"].unique()))
    sel_age     = st.multiselect("Age Group", sorted(df["Age_Group"].unique()),   default=sorted(df["Age_Group"].unique()))

dff = df[
    df["Country"].isin(sel_country) &
    df["Gender"].isin(sel_gender) &
    df["Age_Group"].isin(sel_age)
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<span class="pg-title">Overview</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">2,000 synthetic Hajj pilgrims · 10 countries · 9 variables</p>', unsafe_allow_html=True)

    n      = len(dff)
    avg_sp = dff["Estimated_Spending_SAR"].mean()
    avg_st = dff["Stay_Days"].mean()
    avg_gr = dff["Group_Size"].mean()
    top_c  = dff["Country"].value_counts().idxmax()

    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item">
            <div class="s-label">Pilgrims</div>
            <div class="s-val">{n:,}</div>
            <div class="s-note">filtered</div>
        </div>
        <div class="stat-item">
            <div class="s-label">Avg Spending</div>
            <div class="s-val">{avg_sp:,.0f}</div>
            <div class="s-note">SAR</div>
        </div>
        <div class="stat-item">
            <div class="s-label">Avg Stay</div>
            <div class="s-val">{avg_st:.1f}</div>
            <div class="s-note">days</div>
        </div>
        <div class="stat-item">
            <div class="s-label">Avg Group</div>
            <div class="s-val">{avg_gr:.1f}</div>
            <div class="s-note">persons</div>
        </div>
        <div class="stat-item">
            <div class="s-label">Top Country</div>
            <div class="s-val" style="font-size:1rem;padding-top:5px">{top_c}</div>
            <div class="s-note">most pilgrims</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        vc = dff["Country"].value_counts().reset_index()
        vc.columns = ["Country","Count"]
        f1 = px.bar(vc, x="Country", y="Count", color_discrete_sequence=["#D95F0A"])
        f1.update_traces(marker_line_width=0)
        f1.update_layout(xaxis_tickangle=-35)
        chart(f1, "Pilgrim Count by Country")
    with c2:
        gd = dff["Gender"].value_counts().reset_index()
        gd.columns = ["Gender","Count"]
        f2 = px.pie(gd, names="Gender", values="Count",
                    color_discrete_sequence=["#D95F0A","#F4A96A"], hole=0.5)
        f2.update_traces(textfont_size=12)
        chart(f2, "Gender Distribution")

    c3, c4 = st.columns(2)
    with c3:
        ag = dff["Age_Group"].value_counts().reindex(["18-30","31-45","46-60","60+"]).reset_index()
        ag.columns = ["Age_Group","Count"]
        f3 = px.bar(ag, x="Age_Group", y="Count", color_discrete_sequence=["#D95F0A"])
        f3.update_traces(marker_line_width=0)
        chart(f3, "Pilgrim Count by Age Group")
    with c4:
        ac = dff["Accommodation_Type"].value_counts().reset_index()
        ac.columns = ["Accommodation","Count"]
        f4 = px.pie(ac, names="Accommodation", values="Count",
                    color_discrete_sequence=ORANGE_SEQ, hole=0.45)
        f4.update_traces(textfont_size=11)
        chart(f4, "Accommodation Type Breakdown")

    f5 = px.box(dff, x="Country", y="Estimated_Spending_SAR", color="Country",
                color_discrete_sequence=ORANGE_SEQ,
                labels={"Estimated_Spending_SAR":"Spending (SAR)"})
    f5.update_layout(showlegend=False, xaxis_tickangle=-30)
    chart(f5, "Spending Distribution by Country")

    stacked = dff.groupby(["Country","Transport_Type"]).size().reset_index(name="Count")
    f6 = px.bar(stacked, x="Country", y="Count", color="Transport_Type",
                color_discrete_sequence=ORANGE_SEQ, barmode="stack",
                labels={"Transport_Type":"Transport"})
    f6.update_traces(marker_line_width=0)
    f6.update_layout(xaxis_tickangle=-30)
    chart(f6, "Transport Type by Country")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Descriptive Statistics":
    st.markdown('<span class="pg-title">Descriptive Statistics</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Central tendency, dispersion, shape, and cross-tabulations</p>', unsafe_allow_html=True)

    num_cols = ["Estimated_Spending_SAR","Group_Size","Stay_Days","Spending_Per_Day","Spending_Per_Person"]

    summary = dff[num_cols].describe().T
    summary["Skewness"] = dff[num_cols].skew().round(4)
    summary["Kurtosis"] = dff[num_cols].kurt().round(4)
    summary["Median"]   = dff[num_cols].median().round(2)
    summary["IQR"]      = (dff[num_cols].quantile(0.75) - dff[num_cols].quantile(0.25)).round(2)
    summary["CV (%)"]   = (dff[num_cols].std() / dff[num_cols].mean() * 100).round(2)
    cols_show = ["mean","std","min","25%","50%","75%","max","Median","IQR","Skewness","Kurtosis","CV (%)"]

    st.markdown('<div class="chart-wrap"><div class="chart-title">Numerical Variable Summary</div>', unsafe_allow_html=True)
    st.dataframe(summary[cols_show].round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Distribution Explorer</div>', unsafe_allow_html=True)
    col_sel = st.selectbox("Variable", num_cols)
    d1, d2 = st.columns([3,2])
    with d1:
        fh = px.histogram(dff, x=col_sel, nbins=40, color_discrete_sequence=["#D95F0A"],
                          marginal="box", labels={col_sel: col_sel.replace("_"," ")})
        fh.update_traces(marker_line_width=0)
        st.plotly_chart(T(fh, f"Histogram — {col_sel.replace('_',' ')}"), use_container_width=True)
    with d2:
        fv = px.violin(dff, y=col_sel, x="Gender", color="Gender",
                       color_discrete_sequence=["#D95F0A","#F4A96A"], box=True, points="outliers")
        fv.update_layout(showlegend=False)
        st.plotly_chart(T(fv, "Violin by Gender"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        grp = dff.groupby("Country")["Estimated_Spending_SAR"].mean().reset_index().sort_values("Estimated_Spending_SAR", ascending=False)
        fg = px.bar(grp, x="Country", y="Estimated_Spending_SAR",
                    color_discrete_sequence=["#D95F0A"],
                    labels={"Estimated_Spending_SAR":"Mean Spending (SAR)"})
        fg.update_traces(marker_line_width=0)
        fg.update_layout(xaxis_tickangle=-30)
        chart(fg, "Mean Spending by Country")
    with c2:
        grp2 = dff.groupby("Accommodation_Type")["Estimated_Spending_SAR"].agg(["mean","median","std"]).reset_index()
        grp2.columns = ["Accommodation","Mean","Median","Std Dev"]
        fg2 = go.Figure()
        fg2.add_trace(go.Bar(name="Mean",    x=grp2["Accommodation"], y=grp2["Mean"],    marker_color="#D95F0A", marker_line_width=0))
        fg2.add_trace(go.Bar(name="Median",  x=grp2["Accommodation"], y=grp2["Median"],  marker_color="#F4A96A", marker_line_width=0))
        fg2.add_trace(go.Bar(name="Std Dev", x=grp2["Accommodation"], y=grp2["Std Dev"], marker_color="#FDE0C5", marker_line_width=0))
        fg2.update_layout(barmode="group")
        chart(fg2, "Spending by Accommodation — Mean / Median / Std Dev")

    pivot = dff.pivot_table(values="Estimated_Spending_SAR", index="Country", columns="Age_Group", aggfunc="mean")
    fhm = px.imshow(pivot, color_continuous_scale=[[0,"#FDE0C5"],[0.5,"#F4A96A"],[1,"#A03A04"]],
                    text_auto=".0f", aspect="auto", labels={"color":"Mean SAR"})
    chart(fhm, "Mean Spending Heatmap — Country × Age Group")

    corr = dff[num_cols].corr()
    fco = px.imshow(corr, color_continuous_scale=[[0,"#FDE0C5"],[0.5,"#F4A96A"],[1,"#A03A04"]],
                    text_auto=".2f", zmin=-1, zmax=1, aspect="auto")
    chart(fco, "Correlation Matrix — Numerical Variables")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Confidence Intervals":
    st.markdown('<span class="pg-title">Confidence Intervals & Hypothesis Tests</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Inferential statistics: CIs, t-tests, and ANOVA</p>', unsafe_allow_html=True)

    conf_level = st.slider("Confidence Level (%)", 80, 99, 95) / 100
    alpha = 1 - conf_level

    ci_rows = []
    for country, grp in dff.groupby("Country")["Estimated_Spending_SAR"]:
        n, mean, se = len(grp), grp.mean(), stats.sem(grp)
        tc = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_rows.append({"Country":country,"N":n,"Mean (SAR)":round(mean,2),
                        "SE":round(se,2),"CI Lower":round(mean-tc*se,2),"CI Upper":round(mean+tc*se,2)})
    ci_df = pd.DataFrame(ci_rows).sort_values("Mean (SAR)", ascending=False)

    st.markdown('<div class="chart-wrap"><div class="chart-title">CI Table — Mean Spending by Country</div>', unsafe_allow_html=True)
    st.dataframe(ci_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ffp = go.Figure()
    for _, row in ci_df.iterrows():
        ffp.add_trace(go.Scatter(
            x=[row["CI Lower"], row["Mean (SAR)"], row["CI Upper"]],
            y=[row["Country"]]*3,
            mode="lines+markers",
            line=dict(color="#D95F0A", width=2),
            marker=dict(size=[5,11,5], color=["#F4A96A","#D95F0A","#F4A96A"]),
            name=row["Country"], showlegend=False,
        ))
    ffp.update_layout(xaxis_title="Spending (SAR)", yaxis_title="Country")
    chart(ffp, f"Forest Plot — {int(conf_level*100)}% Confidence Intervals by Country")

    st.markdown("---")

    # One-sample t-test
    st.markdown('<div class="chart-wrap"><div class="chart-title">One-Sample t-Test — Mean Spending vs. Hypothesised Value</div>', unsafe_allow_html=True)
    hyp = st.number_input("Hypothesised Mean (SAR)", value=22000, step=500)
    t1, p1 = stats.ttest_1samp(dff["Estimated_Spending_SAR"], hyp)
    sm_ = dff["Estimated_Spending_SAR"].mean()
    se_ = dff["Estimated_Spending_SAR"].sem()
    tc_ = stats.t.ppf(0.975, df=len(dff)-1)
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Sample Mean",  f"{sm_:,.2f} SAR")
    r2.metric("t-Statistic",  f"{t1:.4f}")
    r3.metric("p-Value",      f"{p1:.4f}")
    r4.metric("Decision",     "Reject H₀" if p1<0.05 else "Fail to Reject H₀")
    st.info(f"95% CI: [{sm_-tc_*se_:,.2f}, {sm_+tc_*se_:,.2f}] SAR  ·  H₀: μ = {hyp:,} SAR")
    st.markdown('</div>', unsafe_allow_html=True)

    # Two-sample t-test
    st.markdown('<div class="chart-wrap"><div class="chart-title">Two-Sample t-Test — Spending by Gender</div>', unsafe_allow_html=True)
    gm = dff[dff["Gender"]=="Male"]["Estimated_Spending_SAR"]
    gf = dff[dff["Gender"]=="Female"]["Estimated_Spending_SAR"]
    t2, p2 = stats.ttest_ind(gm, gf)
    g1, g2 = st.columns([1,2])
    with g1:
        st.dataframe(pd.DataFrame({
            "Group":["Male","Female"],"N":[len(gm),len(gf)],
            "Mean":[round(gm.mean(),2),round(gf.mean(),2)],
            "Std": [round(gm.std(),2), round(gf.std(),2)]
        }), use_container_width=True, hide_index=True)
        st.metric("t-Statistic", f"{t2:.4f}")
        st.metric("p-Value",     f"{p2:.4f}")
        st.info("Significant difference detected" if p2<0.05 else "No significant difference")
    with g2:
        fb = px.box(dff, x="Gender", y="Estimated_Spending_SAR", color="Gender",
                    color_discrete_sequence=["#D95F0A","#F4A96A"], points="outliers",
                    labels={"Estimated_Spending_SAR":"Spending (SAR)"})
        fb.update_layout(showlegend=False)
        st.plotly_chart(T(fb, "Spending by Gender"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ANOVA
    st.markdown('<div class="chart-wrap"><div class="chart-title">One-Way ANOVA — Spending Across Age Groups</div>', unsafe_allow_html=True)
    groups_a = [g["Estimated_Spending_SAR"].values for _,g in dff.groupby("Age_Group")]
    fs, ps = stats.f_oneway(*groups_a)
    a1,a2,a3 = st.columns(3)
    a1.metric("F-Statistic", f"{fs:.4f}")
    a2.metric("p-Value",     f"{ps:.4f}")
    a3.metric("Decision",    "Reject H₀" if ps<0.05 else "Fail to Reject H₀")
    st.info("H₀: Mean spending is equal across all age groups")
    fa = px.box(dff, x="Age_Group", y="Estimated_Spending_SAR", color="Age_Group",
                color_discrete_sequence=ORANGE_SEQ, points="outliers",
                category_orders={"Age_Group":["18-30","31-45","46-60","60+"]},
                labels={"Estimated_Spending_SAR":"Spending (SAR)","Age_Group":"Age Group"})
    fa.update_layout(showlegend=False)
    st.plotly_chart(T(fa, "Spending by Age Group"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PROBABILITY & DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Probability & Distributions":
    st.markdown('<span class="pg-title">Probability & Distributions</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Empirical probabilities, distribution fitting, and normality tests</p>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        pc = (dff["Country"].value_counts(normalize=True)*100).reset_index()
        pc.columns = ["Country","P (%)"]
        pc["P (%)"] = pc["P (%)"].round(2)
        st.markdown('<div class="chart-wrap"><div class="chart-title">P(Country)</div>', unsafe_allow_html=True)
        st.dataframe(pc, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with e2:
        pa = (dff["Accommodation_Type"].value_counts(normalize=True)*100).reset_index()
        pa.columns = ["Accommodation","P (%)"]
        pa["P (%)"] = pa["P (%)"].round(2)
        st.markdown('<div class="chart-wrap"><div class="chart-title">P(Accommodation Type)</div>', unsafe_allow_html=True)
        st.dataframe(pa, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    spend = dff["Estimated_Spending_SAR"].dropna()
    mu, sigma = spend.mean(), spend.std()
    x_r = np.linspace(spend.min(), spend.max(), 300)
    pdf_v = stats.norm.pdf(x_r, mu, sigma)

    fn = go.Figure()
    fn.add_trace(go.Histogram(x=spend, nbinsx=50, histnorm="probability density",
                              name="Empirical", marker_color="#F4A96A",
                              marker_line_width=0, opacity=0.75))
    fn.add_trace(go.Scatter(x=x_r, y=pdf_v, mode="lines",
                            name=f"Normal (μ={mu:.0f}, σ={sigma:.0f})",
                            line=dict(color="#D95F0A", width=2)))
    fn.update_layout(xaxis_title="Spending (SAR)", yaxis_title="Density")
    chart(fn, "Normal Distribution Fit — Estimated Spending (SAR)")

    ks_stat, ks_p = stats.kstest(spend, "norm", args=(mu, sigma))
    sw_stat, sw_p = stats.shapiro(spend.sample(min(5000, len(spend)), random_state=42))
    n1,n2,n3,n4 = st.columns(4)
    n1.metric("μ (Mean)",      f"{mu:,.2f} SAR")
    n2.metric("σ (Std Dev)",   f"{sigma:,.2f} SAR")
    n3.metric("K-S p-value",   f"{ks_p:.4f}")
    n4.metric("Shapiro-Wilk p",f"{sw_p:.4f}")
    st.info("Data is approximately normal" if ks_p > 0.05 else "Data deviates from normality (K-S test)")

    (osm, osr), (slope, intercept, _) = stats.probplot(spend, dist="norm")
    fq = go.Figure()
    fq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Observed",
                            marker=dict(color="#D95F0A", size=4, opacity=0.5)))
    fq.add_trace(go.Scatter(x=[min(osm),max(osm)],
                            y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
                            mode="lines", name="Reference",
                            line=dict(color="#888888", width=1.5, dash="dash")))
    fq.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    chart(fq, "Q-Q Plot — Spending vs. Normal Distribution")

    lam = dff["Group_Size"].mean()
    k_vals = np.arange(0, int(dff["Group_Size"].max())+2)
    pmf = stats.poisson.pmf(k_vals, lam)
    emp = dff["Group_Size"].value_counts(normalize=True).reindex(k_vals, fill_value=0)
    fp = go.Figure()
    fp.add_trace(go.Bar(x=k_vals, y=emp.values, name="Empirical",
                        marker_color="#F4A96A", marker_line_width=0, opacity=0.75))
    fp.add_trace(go.Scatter(x=k_vals, y=pmf, mode="lines+markers",
                            name=f"Poisson (λ={lam:.2f})",
                            line=dict(color="#D95F0A", width=2),
                            marker=dict(size=7, color="#D95F0A")))
    fp.update_layout(xaxis_title="Group Size", yaxis_title="Probability")
    chart(fp, f"Poisson Distribution Fit — Group Size (λ = {lam:.2f})")

    st.markdown('<div class="chart-wrap"><div class="chart-title">Conditional Probability — P(High Spender | Accommodation Type)</div>', unsafe_allow_html=True)
    threshold = st.slider("High Spender threshold (SAR)", 15000, 35000, 25000, step=1000)
    dff2 = dff.copy()
    dff2["HS"] = dff2["Estimated_Spending_SAR"] >= threshold
    cp = dff2.groupby("Accommodation_Type")["HS"].mean().mul(100).reset_index()
    cp.columns = ["Accommodation","P (%)"]
    fcp = px.bar(cp.sort_values("P (%)", ascending=False), x="Accommodation", y="P (%)",
                 color_discrete_sequence=["#D95F0A"], text_auto=".1f")
    fcp.update_traces(marker_line_width=0)
    fcp.update_layout(yaxis_title="Probability (%)")
    st.plotly_chart(T(fcp, f"P(Spending ≥ {threshold:,} SAR | Accommodation)"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — REGRESSION & PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Regression & Predictions":
    st.markdown('<span class="pg-title">Regression & Predictions</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Simple linear, multiple linear regression, and spending predictor</p>', unsafe_allow_html=True)

    dfe = dff.copy()
    dfe["Gender_Num"]    = (dfe["Gender"]=="Male").astype(int)
    dfe["Accom_Num"]     = dfe["Accommodation_Type"].map({"Camp (Mina)":0,"Apartment":1,"Hotel (3 Star)":2,"Hotel (5 Star)":3})
    dfe["Age_Num"]       = dfe["Age_Group"].map({"18-30":1,"31-45":2,"46-60":3,"60+":4})
    dfe["Transport_Num"] = dfe["Transport_Type"].map({"Bus":0,"Group Transport":1,"Train (Haramain)":2,"Private Car":3})
    feat_cols   = ["Stay_Days","Group_Size","Accom_Num","Age_Num","Transport_Num","Gender_Num"]
    feat_labels = ["Stay Days","Group Size","Accommodation","Age Group","Transport","Gender"]

    # SLR
    X_slr   = sm.add_constant(dfe["Stay_Days"])
    mdl_slr = sm.OLS(dfe["Estimated_Spending_SAR"], X_slr).fit()
    sl, ic  = mdl_slr.params["Stay_Days"], mdl_slr.params["const"]
    r2s     = mdl_slr.rsquared

    fsl = px.scatter(dfe, x="Stay_Days", y="Estimated_Spending_SAR",
                     color="Accommodation_Type", color_discrete_sequence=ORANGE_SEQ,
                     opacity=0.4, labels={"Stay_Days":"Stay Duration (Days)","Estimated_Spending_SAR":"Spending (SAR)"})
    xl = np.array([dfe["Stay_Days"].min(), dfe["Stay_Days"].max()])
    fsl.add_trace(go.Scatter(x=xl, y=sl*xl+ic, mode="lines", name="Regression Line",
                             line=dict(color="#111111", width=2, dash="dash")))
    fsl.update_layout(legend_title="Accommodation")
    chart(fsl, f"Simple Linear Regression — Stay Days vs. Spending  (R² = {r2s:.4f})")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Intercept",       f"{ic:,.2f}")
    m2.metric("Slope",           f"{sl:,.2f}")
    m3.metric("R²",              f"{r2s:.4f}")
    m4.metric("p-value (slope)", f"{mdl_slr.pvalues['Stay_Days']:.4e}")

    st.markdown("---")

    # MLR
    X_mlr   = sm.add_constant(dfe[feat_cols])
    mdl_mlr = sm.OLS(dfe["Estimated_Spending_SAR"], X_mlr).fit()

    coef_df = pd.DataFrame({
        "Feature":     feat_labels,
        "Coefficient": mdl_mlr.params[feat_cols].values.round(4),
        "Std Error":   mdl_mlr.bse[feat_cols].values.round(4),
        "t-Stat":      mdl_mlr.tvalues[feat_cols].values.round(4),
        "p-Value":     mdl_mlr.pvalues[feat_cols].values.round(4),
        "Significant": mdl_mlr.pvalues[feat_cols].values < 0.05,
    })

    g1, g2 = st.columns([2,3])
    with g1:
        st.markdown('<div class="chart-wrap"><div class="chart-title">MLR Coefficients</div>', unsafe_allow_html=True)
        st.dataframe(coef_df.drop("Significant", axis=1), use_container_width=True, hide_index=True)
        st.metric("Adj. R²",     f"{mdl_mlr.rsquared_adj:.4f}")
        st.metric("F-Statistic", f"{mdl_mlr.fvalue:.2f}")
        st.metric("AIC",         f"{mdl_mlr.aic:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with g2:
        colors = ["#D95F0A" if s else "#CCCCCC" for s in coef_df["Significant"]]
        fc = go.Figure()
        fc.add_trace(go.Bar(
            x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
            marker_color=colors, marker_line_width=0,
            error_x=dict(array=coef_df["Std Error"], color="#888888", thickness=1.5)
        ))
        fc.update_layout(xaxis_title="Coefficient", yaxis_title="")
        chart(fc, "Coefficient Plot  (orange = significant at α = 0.05)")

    preds  = mdl_mlr.fittedvalues
    resids = mdl_mlr.resid
    rv1, rv2 = st.columns(2)
    with rv1:
        frv = px.scatter(x=preds, y=resids, opacity=0.3, color_discrete_sequence=["#D95F0A"],
                         labels={"x":"Fitted (SAR)","y":"Residuals (SAR)"})
        frv.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1)
        frv.update_traces(marker_size=4)
        chart(frv, "Residuals vs. Fitted Values")
    with rv2:
        frh = px.histogram(x=resids, nbins=40, color_discrete_sequence=["#D95F0A"],
                           labels={"x":"Residuals (SAR)","y":"Count"})
        frh.update_traces(marker_line_width=0)
        chart(frh, "Residual Distribution")

    st.markdown("---")

    # Predictor
    st.markdown('<div class="chart-wrap"><div class="chart-title">Spending Predictor</div>', unsafe_allow_html=True)
    st.caption("Enter pilgrim details to predict estimated spending.")
    p1, p2, p3 = st.columns(3)
    with p1:
        p_stay  = st.selectbox("Stay Duration (days)", [7,10,14,21], index=1)
        p_group = st.slider("Group Size", 1, 5, 2)
    with p2:
        p_accom = st.selectbox("Accommodation", ["Camp (Mina)","Apartment","Hotel (3 Star)","Hotel (5 Star)"])
        p_age   = st.selectbox("Age Group", ["18-30","31-45","46-60","60+"])
    with p3:
        p_trans  = st.selectbox("Transport", ["Bus","Group Transport","Train (Haramain)","Private Car"])
        p_gender = st.selectbox("Gender", ["Male","Female"])

    if st.button("Predict Spending"):
        new_x = pd.DataFrame([[
            1, p_stay, p_group,
            {"Camp (Mina)":0,"Apartment":1,"Hotel (3 Star)":2,"Hotel (5 Star)":3}[p_accom],
            {"18-30":1,"31-45":2,"46-60":3,"60+":4}[p_age],
            {"Bus":0,"Group Transport":1,"Train (Haramain)":2,"Private Car":3}[p_trans],
            int(p_gender=="Male")
        ]], columns=["const"]+feat_cols)
        pv = mdl_mlr.predict(new_x)[0]
        pl = pv - 1.96*np.sqrt(mdl_mlr.mse_resid)
        ph = pv + 1.96*np.sqrt(mdl_mlr.mse_resid)
        st.success(f"Predicted Spending: **{pv:,.0f} SAR**")
        st.info(f"95% Prediction Interval: {pl:,.0f} — {ph:,.0f} SAR")

        fg_ = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pv,
            title={"text":"Predicted Spending (SAR)","font":{"size":13,"family":"IBM Plex Sans"}},
            delta={"reference": dff["Estimated_Spending_SAR"].mean(), "valueformat":",.0f"},
            gauge={
                "axis":{"range":[5000,40000],"tickformat":",.0f","tickfont":{"size":10}},
                "bar": {"color":"#D95F0A","thickness":0.22},
                "bgcolor":"#FFFFFF",
                "bordercolor":"#E2E2E2",
                "steps":[
                    {"range":[5000,15000], "color":"#FDE0C5"},
                    {"range":[15000,25000],"color":"#F9C99A"},
                    {"range":[25000,35000],"color":"#F4A96A"},
                    {"range":[35000,40000],"color":"#E8883A"},
                ],
                "threshold":{"line":{"color":"#888888","width":2},"thickness":0.75,
                             "value":dff["Estimated_Spending_SAR"].mean()},
            }
        ))
        fg_.update_layout(paper_bgcolor="#FFFFFF", font_family="IBM Plex Sans",
                          height=260, margin=dict(l=30,r=30,t=40,b=20))
        st.plotly_chart(fg_, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Raw Data":
    st.markdown('<span class="pg-title">Raw Data</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Browse, filter, search, sort, and export the dataset</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item"><div class="s-label">Rows</div><div class="s-val">{len(dff):,}</div><div class="s-note">filtered</div></div>
        <div class="stat-item"><div class="s-label">Columns</div><div class="s-val">{len(dff.columns)}</div><div class="s-note">variables</div></div>
        <div class="stat-item"><div class="s-label">Missing</div><div class="s-val">0</div><div class="s-note">complete</div></div>
        <div class="stat-item"><div class="s-label">Numeric</div><div class="s-val">5</div><div class="s-note">incl. derived</div></div>
        <div class="stat-item"><div class="s-label">Categorical</div><div class="s-val">6</div><div class="s-note">original</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Column Descriptions"):
        st.dataframe(pd.DataFrame({
            "Column": ["Pilgrim_ID","Country","Gender","Age_Group","Accommodation_Type",
                       "Transport_Type","Stay_Duration","Estimated_Spending_SAR","Group_Size",
                       "Stay_Days","Spending_Per_Day","Spending_Per_Person"],
            "Type": ["ID","Categorical","Categorical","Categorical","Categorical",
                     "Categorical","Categorical","Numerical","Numerical",
                     "Derived","Derived","Derived"],
            "Description": [
                "Unique pilgrim identifier (1–2000)",
                "Country of origin — 10 countries",
                "Male or Female",
                "Age bracket: 18-30, 31-45, 46-60, 60+",
                "Camp (Mina), Apartment, Hotel 3★, Hotel 5★",
                "Bus, Group Transport, Train (Haramain), Private Car",
                "Duration of stay: 7, 10, 14, or 21 days",
                "Total estimated spending in Saudi Riyals",
                "Number of persons in travel group (1–5)",
                "Stay_Duration parsed as integer number of days",
                "Spending ÷ Stay_Days",
                "Spending ÷ Group_Size",
            ]
        }), use_container_width=True, hide_index=True)

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([3,1,1])
    with ctrl1:
        search_text = st.text_input("Search across all columns", placeholder="e.g. Pakistan, Hotel, Female…")
    with ctrl2:
        sort_col = st.selectbox("Sort by", ["Pilgrim_ID","Estimated_Spending_SAR","Stay_Days","Group_Size"])
    with ctrl3:
        sort_asc = st.checkbox("Ascending", value=True)

    all_cols = list(dff.columns)
    sel_cols = st.multiselect("Columns to display", all_cols, default=all_cols[:9])

    display_df = dff[sel_cols].copy()
    if search_text.strip():
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_text.strip(), case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-title">Dataset — {len(display_df):,} rows</div>', unsafe_allow_html=True)
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        label="Download Filtered Dataset (.csv)",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="hajj_filtered.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown('<div class="sec-label">Distribution Snapshots</div>', unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        fq1 = px.histogram(dff, x="Estimated_Spending_SAR", nbins=30, color_discrete_sequence=["#D95F0A"])
        fq1.update_traces(marker_line_width=0)
        chart(fq1, "Spending (SAR)")
    with qc2:
        fq2 = px.histogram(dff, x="Group_Size", nbins=5, color_discrete_sequence=["#E8883A"])
        fq2.update_traces(marker_line_width=0)
        chart(fq2, "Group Size")
    with qc3:
        fq3 = px.histogram(dff, x="Stay_Days", nbins=4, color_discrete_sequence=["#F4A96A"])
        fq3.update_traces(marker_line_width=0)
        chart(fq3, "Stay Duration (days)")