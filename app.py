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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HalalStats — Hajj Analytics",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

.block-container { padding: 32px 44px 64px 44px !important; max-width: 1180px !important; }

.pg-title {
    font-size: 1.4rem; font-weight: 600; color: var(--black);
    border-bottom: 2px solid var(--orange);
    padding-bottom: 8px; margin-bottom: 6px; display: inline-block;
}
.pg-sub { font-size: 12.5px; color: var(--grey-2); margin-bottom: 28px; }

.sec-label {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--grey-2); margin-bottom: 8px;
}

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

.chart-wrap {
    border: 1px solid var(--border); border-radius: 4px;
    padding: 18px 20px; margin-bottom: 18px; background: var(--white);
}
.chart-title { font-size: 12.5px; font-weight: 600; color: var(--black); margin-bottom: 12px; }

hr { border-color: var(--border) !important; margin: 24px 0 !important; }

.dataframe thead tr th {
    background: var(--grey-4) !important; color: var(--black) !important;
    font-weight: 600 !important; font-size: 11.5px !important;
}
.dataframe tbody tr:nth-child(even) { background: #FAFAFA !important; }
.dataframe { font-size: 12px !important; }

.stButton > button {
    background: var(--orange) !important; color: var(--white) !important;
    border: none !important; border-radius: 3px !important;
    font-weight: 500 !important; font-size: 13px !important;
    padding: 8px 22px !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

[data-baseweb="select"] > div {
    border-color: var(--border) !important; border-radius: 3px !important;
    font-size: 13px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--orange) !important; font-family: 'IBM Plex Mono', monospace !important;
}

.stAlert { border-radius: 3px !important; font-size: 12.5px !important; }
[data-testid="stInfo"] {
    border-left: 3px solid var(--orange) !important;
    background: var(--orange-3) !important;
}
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 3px !important; }

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
    df["Stay_Days"] = df["Stay_Duration"].str.extract(r"(\d+)").astype(int)
    # REMOVED: Spending_Per_Day and Spending_Per_Person — derived cols not in dataset/outline
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
        "Probability & Distributions",
        "Confidence Intervals",
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
# Week 1: Bar chart, Pie chart, Freq dist for qualitative data,
#          quantitative/discrete freq dist, C.F dist
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<span class="pg-title">Overview</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Graphical and tabular representation of Hajj pilgrim data — 2,000 pilgrims · 10 countries · 9 variables  |  Week 1</p>', unsafe_allow_html=True)

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

    # ── Qualitative Frequency Distribution Table (Week 1) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Frequency Distribution Table — Country (Qualitative Data)  |  Week 1</div>', unsafe_allow_html=True)
    freq_country = dff["Country"].value_counts().reset_index()
    freq_country.columns = ["Country", "Frequency"]
    freq_country["Relative Frequency"] = (freq_country["Frequency"] / freq_country["Frequency"].sum()).round(4)
    freq_country["Cumulative Frequency"] = freq_country["Frequency"].cumsum()
    freq_country["Cumulative RF"] = freq_country["Relative Frequency"].cumsum().round(4)
    st.dataframe(freq_country, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quantitative / Discrete Frequency Distribution (Week 1) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Frequency Distribution Table — Group Size (Quantitative Discrete Data)  |  Week 1</div>', unsafe_allow_html=True)
    freq_gs = dff["Group_Size"].value_counts().sort_index().reset_index()
    freq_gs.columns = ["Group Size", "Frequency"]
    freq_gs["Relative Frequency"] = (freq_gs["Frequency"] / freq_gs["Frequency"].sum()).round(4)
    freq_gs["Cumulative Frequency"] = freq_gs["Frequency"].cumsum()
    freq_gs["Cumulative RF"] = freq_gs["Relative Frequency"].cumsum().round(4)
    st.dataframe(freq_gs, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bar chart (Week 1) ──
    c1, c2 = st.columns(2)
    with c1:
        vc = dff["Country"].value_counts().reset_index()
        vc.columns = ["Country","Frequency"]
        f1 = px.bar(vc, x="Country", y="Frequency", color_discrete_sequence=["#D95F0A"],
                    labels={"Frequency":"Number of Pilgrims"})
        f1.update_traces(marker_line_width=0)
        f1.update_layout(xaxis_tickangle=-35)
        chart(f1, "Bar Chart — Pilgrim Count by Country  |  Week 1")

    # ── Pie chart (Week 1) ──
    with c2:
        gd = dff["Gender"].value_counts().reset_index()
        gd.columns = ["Gender","Count"]
        f2 = px.pie(gd, names="Gender", values="Count",
                    color_discrete_sequence=["#D95F0A","#F4A96A"], hole=0.5)
        f2.update_traces(textfont_size=12)
        chart(f2, "Pie Chart — Gender Distribution  |  Week 1")

    c3, c4 = st.columns(2)
    with c3:
        ag = dff["Age_Group"].value_counts().reindex(["18-30","31-45","46-60","60+"]).reset_index()
        ag.columns = ["Age_Group","Frequency"]
        f3 = px.bar(ag, x="Age_Group", y="Frequency", color_discrete_sequence=["#D95F0A"],
                    labels={"Age_Group":"Age Group","Frequency":"Number of Pilgrims"})
        f3.update_traces(marker_line_width=0)
        chart(f3, "Bar Chart — Pilgrim Count by Age Group  |  Week 1")

    with c4:
        ac = dff["Accommodation_Type"].value_counts().reset_index()
        ac.columns = ["Accommodation","Count"]
        f4 = px.pie(ac, names="Accommodation", values="Count",
                    color_discrete_sequence=ORANGE_SEQ, hole=0.45)
        f4.update_traces(textfont_size=11)
        chart(f4, "Pie Chart — Accommodation Type  |  Week 1")

    # NOTE: Stacked bar chart REMOVED — not specified in Week 1 (only Bar & Pie charts listed)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DESCRIPTIVE STATISTICS
# Week 2: Mean, Median, Mode, Trimmed Mean, Quartile, Percentile,
#          Variance, Std Dev, CV, IQR, Five-point summary, Box-plot
# Week 3: Grouped data (Freq, PF, RF, CF, Mean, Variance), Dot Plot, Histogram
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Descriptive Statistics":
    st.markdown('<span class="pg-title">Descriptive Statistics</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Measures of central tendency and dispersion — ungrouped (Week 2) and grouped data (Week 3)</p>', unsafe_allow_html=True)

    # Only columns from the actual dataset (no derived spending_per_day etc.)
    num_cols = ["Estimated_Spending_SAR", "Group_Size", "Stay_Days"]

    # ── Five-point summary + descriptive measures (Week 2) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Descriptive Measures — Ungrouped Data  |  Week 2</div>', unsafe_allow_html=True)
    rows = []
    for col in num_cols:
        s = dff[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        # Trimmed mean (Week 2 explicitly lists it)
        trimmed = stats.trim_mean(s, 0.05)
        rows.append({
            "Variable":        col.replace("_"," "),
            "Min":             round(s.min(), 2),
            "Q1 (25%)":        round(q1, 2),
            "Median":          round(s.median(), 2),
            "Q3 (75%)":        round(q3, 2),
            "Max":             round(s.max(), 2),
            "Mean":            round(s.mean(), 2),
            "Trimmed Mean (5%)": round(trimmed, 2),
            "Mode":            round(s.mode()[0], 2),
            "Std Dev":         round(s.std(), 2),
            "Variance":        round(s.var(), 2),
            "IQR":             round(q3 - q1, 2),
            "CV (%)":          round(s.std() / s.mean() * 100, 2),
            "P10":             round(s.quantile(0.10), 2),
            "P90":             round(s.quantile(0.90), 2),
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.info("Five-point summary: Min, Q1, Median, Q3, Max. Trimmed mean removes extreme 5% on each end before averaging.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Box plot — Five-point summary visual (Week 2) ──
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Box Plot — Five-Point Summary Visualised  |  Week 2</div>', unsafe_allow_html=True)
    col_sel = st.selectbox("Select variable", num_cols, key="bp_sel")
    bp1, bp2 = st.columns([3, 2])
    with bp1:
        fb = px.box(dff, y=col_sel, x="Country", color="Country",
                    color_discrete_sequence=ORANGE_SEQ, points="outliers",
                    labels={col_sel: col_sel.replace("_"," "), "Country":"Country"})
        fb.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(T(fb, f"Box Plot — {col_sel.replace('_',' ')} by Country"), use_container_width=True)
    with bp2:
        # ── Histogram (Week 3) ──
        fh = px.histogram(dff, x=col_sel, nbins=30, color_discrete_sequence=["#D95F0A"],
                          labels={col_sel: col_sel.replace("_"," ")})
        fh.update_traces(marker_line_width=0)
        st.plotly_chart(T(fh, f"Histogram — {col_sel.replace('_',' ')}  |  Week 3"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Dot plot (Week 3) ──
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Dot Plot — Group Size Distribution  |  Week 3</div>', unsafe_allow_html=True)
    dot_counts = dff["Group_Size"].value_counts().sort_index().reset_index()
    dot_counts.columns = ["Group_Size","Count"]
    fdot = px.scatter(dot_counts, x="Group_Size", y="Count",
                      color_discrete_sequence=["#D95F0A"], size="Count",
                      labels={"Group_Size":"Group Size","Count":"Number of Pilgrims"})
    fdot.update_traces(marker_line_width=1, marker_line_color="#C04A05")
    st.plotly_chart(T(fdot, "Dot Plot — Frequency of Each Group Size"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Grouped data: Freq, PF, RF, CF, Mean, Variance (Week 3) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Grouped Data — Frequency Distribution of Spending (SAR)  |  Week 3</div>', unsafe_allow_html=True)
    spend = dff["Estimated_Spending_SAR"]
    bins  = pd.cut(spend, bins=8)
    freq_table = bins.value_counts(sort=False).reset_index()
    freq_table.columns = ["Class Interval","Frequency"]
    freq_table["Class Interval"] = freq_table["Class Interval"].astype(str)
    total_n = freq_table["Frequency"].sum()
    freq_table["P.F (Proportion Freq)"] = (freq_table["Frequency"] / total_n).round(4)
    freq_table["R.F (Relative Freq)"]   = freq_table["P.F (Proportion Freq)"]   # same as PF
    freq_table["C.F (Cumulative Freq)"] = freq_table["Frequency"].cumsum()

    # Grouped mean and variance using midpoints
    intervals = pd.cut(spend, bins=8, retbins=True)
    bin_edges = intervals[1]
    midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    freq_vals  = freq_table["Frequency"].values
    grp_mean   = round(np.sum(np.array(midpoints) * freq_vals) / total_n, 2)
    grp_var    = round(np.sum(freq_vals * (np.array(midpoints) - grp_mean)**2) / (total_n - 1), 2)

    st.dataframe(freq_table, use_container_width=True, hide_index=True)
    gm1, gm2 = st.columns(2)
    gm1.metric("Grouped Mean (SAR)",     f"{grp_mean:,}")
    gm2.metric("Grouped Variance (SAR²)", f"{grp_var:,}")
    st.info("P.F = Proportion Frequency = f/n · R.F = Relative Frequency · C.F = Cumulative Frequency. Grouped mean = Σ(midpoint × f) / n.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Mean spending grouped bar (kept — uses Week 2/3 concepts) ──
    c1, c2 = st.columns(2)
    with c1:
        grp = dff.groupby("Country")["Estimated_Spending_SAR"].mean().reset_index().sort_values("Estimated_Spending_SAR", ascending=False)
        fg = px.bar(grp, x="Country", y="Estimated_Spending_SAR",
                    color_discrete_sequence=["#D95F0A"],
                    labels={"Estimated_Spending_SAR":"Mean Spending (SAR)"})
        fg.update_traces(marker_line_width=0)
        fg.update_layout(xaxis_tickangle=-30)
        chart(fg, "Bar Chart — Mean Spending by Country  |  Week 2")
    with c2:
        grp2 = dff.groupby("Accommodation_Type")["Estimated_Spending_SAR"].agg(["mean","median"]).reset_index()
        grp2.columns = ["Accommodation","Mean","Median"]
        fg2 = go.Figure()
        fg2.add_trace(go.Bar(name="Mean",   x=grp2["Accommodation"], y=grp2["Mean"],   marker_color="#D95F0A", marker_line_width=0))
        fg2.add_trace(go.Bar(name="Median", x=grp2["Accommodation"], y=grp2["Median"], marker_color="#F4A96A", marker_line_width=0))
        fg2.update_layout(barmode="group")
        chart(fg2, "Grouped Bar Chart — Mean vs Median Spending by Accommodation  |  Week 2")

    # NOTE: Cross-tabulation heatmap (Country × Age Group) REMOVED —
    # heatmaps are not listed in Week 1–3 graphical techniques (only Bar, Pie, Dot Plot, Histogram)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PROBABILITY & DISTRIBUTIONS
# Week 4: Introduction to probability, counting techniques,
#          cross tab and joint probability table
# Week 5: Probability of an event, addition law, Conditional Probability
# Week 6: Independence, Multiplicative rules, Bayes rule
# Week 10: Binomial, Hypergeometric and Poisson Distribution
# Week 11: Normal distribution, Area under normal curve, Standard Normal
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Probability & Distributions":
    st.markdown('<span class="pg-title">Probability & Distributions</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Empirical probability, conditional probability, Bayes rule, Poisson and Normal distributions  |  Weeks 4–6, 10–11</p>', unsafe_allow_html=True)

    # ── Empirical probability tables (Week 4–5) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Empirical Probability — Frequency & Relative Frequency Tables  |  Week 4</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        pc = dff["Country"].value_counts().reset_index()
        pc.columns = ["Country","Frequency"]
        pc["P(Country)"] = (pc["Frequency"] / pc["Frequency"].sum()).round(4)
        st.caption("P(Pilgrim is from each Country)")
        st.dataframe(pc, use_container_width=True, hide_index=True)
    with e2:
        pa = dff["Accommodation_Type"].value_counts().reset_index()
        pa.columns = ["Accommodation","Frequency"]
        pa["P(Accommodation)"] = (pa["Frequency"] / pa["Frequency"].sum()).round(4)
        st.caption("P(Pilgrim uses each Accommodation Type)")
        st.dataframe(pa, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Joint probability / Cross-tab table (Week 4) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Joint Probability Table — Gender × Accommodation Type  |  Week 4</div>', unsafe_allow_html=True)
    joint = pd.crosstab(dff["Gender"], dff["Accommodation_Type"], normalize=True).round(4)
    joint["Row Total"] = joint.sum(axis=1).round(4)
    col_totals = joint.sum(axis=0).round(4)
    col_totals.name = "Column Total"
    joint = pd.concat([joint, col_totals.to_frame().T])
    st.dataframe(joint, use_container_width=True)
    st.info("Each cell = P(Gender AND Accommodation). Row/column totals = marginal probabilities. This is the cross-tabulation and joint probability table (Week 4).")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Addition Law (Week 5) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Addition Law — P(A ∪ B)  |  Week 5</div>', unsafe_allow_html=True)
    al1, al2 = st.columns(2)
    with al1:
        st.caption("Select two events to apply the Addition Law: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)")
        event_a_col = st.selectbox("Event A — Variable", ["Gender", "Accommodation_Type", "Age_Group"], key="al_a_col")
        event_a_val = st.selectbox("Event A — Value", sorted(dff[event_a_col].unique()), key="al_a_val")
    with al2:
        event_b_col = st.selectbox("Event B — Variable", ["Accommodation_Type", "Gender", "Age_Group"], key="al_b_col")
        event_b_val = st.selectbox("Event B — Value", sorted(dff[event_b_col].unique()), key="al_b_val")

    n_total = len(dff)
    p_a   = (dff[event_a_col] == event_a_val).sum() / n_total
    p_b   = (dff[event_b_col] == event_b_val).sum() / n_total
    p_ab  = ((dff[event_a_col] == event_a_val) & (dff[event_b_col] == event_b_val)).sum() / n_total
    p_aub = p_a + p_b - p_ab

    add_cols = st.columns(4)
    add_cols[0].metric("P(A)",       f"{p_a:.4f}")
    add_cols[1].metric("P(B)",       f"{p_b:.4f}")
    add_cols[2].metric("P(A ∩ B)",   f"{p_ab:.4f}")
    add_cols[3].metric("P(A ∪ B)",   f"{p_aub:.4f}")
    st.info(f"P(A ∪ B) = P(A) + P(B) − P(A ∩ B) = {p_a:.4f} + {p_b:.4f} − {p_ab:.4f} = **{p_aub:.4f}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Conditional Probability (Week 5) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Conditional Probability — P(High Spender | Accommodation Type)  |  Week 5</div>', unsafe_allow_html=True)
    threshold = st.slider("Define 'High Spender' threshold (SAR)", 15000, 35000, 25000, step=1000)
    dff2 = dff.copy()
    dff2["High_Spender"] = dff2["Estimated_Spending_SAR"] >= threshold
    cp = dff2.groupby("Accommodation_Type")["High_Spender"].mean().mul(100).reset_index()
    cp.columns = ["Accommodation","P (%)"]
    fcp = px.bar(cp.sort_values("P (%)", ascending=False), x="Accommodation", y="P (%)",
                 color_discrete_sequence=["#D95F0A"], text_auto=".1f",
                 labels={"P (%)":"Probability (%)"})
    fcp.update_traces(marker_line_width=0)
    st.plotly_chart(T(fcp, f"P(Spending ≥ {threshold:,} SAR | Accommodation Type)"), use_container_width=True)
    st.info("P(B|A) = P(A ∩ B) / P(A). Here B = High Spender event, A = Accommodation Type.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Independence & Multiplicative Rule (Week 6) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Independence & Multiplicative Rule  |  Week 6</div>', unsafe_allow_html=True)
    st.caption("Two events A and B are independent if P(A ∩ B) = P(A) · P(B). Check below:")
    ind1, ind2 = st.columns(2)
    with ind1:
        i_a_col = st.selectbox("Event A", ["Gender"], key="ind_a")
        i_a_val = st.selectbox("A = ", sorted(dff[i_a_col].unique()), key="ind_av")
    with ind2:
        i_b_col = st.selectbox("Event B", ["Accommodation_Type"], key="ind_b")
        i_b_val = st.selectbox("B = ", sorted(dff[i_b_col].unique()), key="ind_bv")

    ip_a   = (dff[i_a_col] == i_a_val).sum() / len(dff)
    ip_b   = (dff[i_b_col] == i_b_val).sum() / len(dff)
    ip_ab  = ((dff[i_a_col] == i_a_val) & (dff[i_b_col] == i_b_val)).sum() / len(dff)
    ip_axb = ip_a * ip_b
    is_independent = abs(ip_ab - ip_axb) < 0.01

    ic = st.columns(3)
    ic[0].metric("P(A) · P(B)",   f"{ip_axb:.4f}")
    ic[1].metric("P(A ∩ B)",      f"{ip_ab:.4f}")
    ic[2].metric("Independent?",  "≈ Yes" if is_independent else "No")

    if is_independent:
        st.success(f"P(A ∩ B) ≈ P(A) · P(B) → Events are approximately independent.")
    else:
        st.warning(f"P(A ∩ B) ≠ P(A) · P(B) → Events are NOT independent (difference = {abs(ip_ab - ip_axb):.4f}).")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bayes' Rule (Week 6) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Bayes\' Rule — P(Country | High Spender)  |  Week 6</div>', unsafe_allow_html=True)
    st.caption("Bayes' Theorem: P(A|B) = P(B|A) · P(A) / P(B)")

    bayes_thresh = st.slider("High Spender threshold for Bayes (SAR)", 15000, 35000, 25000, step=1000, key="bayes_thresh")
    dff3 = dff.copy()
    dff3["High_Spender"] = dff3["Estimated_Spending_SAR"] >= bayes_thresh

    p_hs = dff3["High_Spender"].mean()  # P(B) = P(High Spender)
    bayes_rows = []
    for country in sorted(dff3["Country"].unique()):
        mask_c  = dff3["Country"] == country
        p_c     = mask_c.sum() / len(dff3)                        # P(A) = P(Country)
        p_hs_c  = dff3.loc[mask_c, "High_Spender"].mean()         # P(B|A) = P(HS | Country)
        p_c_hs  = (p_hs_c * p_c) / p_hs if p_hs > 0 else 0       # P(A|B) = Bayes
        bayes_rows.append({
            "Country":          country,
            "P(Country)":       round(p_c, 4),
            "P(HS|Country)":    round(p_hs_c, 4),
            "P(Country|HS)":    round(p_c_hs, 4),
        })

    bayes_df = pd.DataFrame(bayes_rows).sort_values("P(Country|HS)", ascending=False)
    fb_bar = px.bar(bayes_df, x="Country", y="P(Country|HS)",
                    color_discrete_sequence=["#D95F0A"], text_auto=".3f",
                    labels={"P(Country|HS)": "P(Country | High Spender)"})
    fb_bar.update_traces(marker_line_width=0)
    fb_bar.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(T(fb_bar, f"Bayes' Rule — P(Country | Spending ≥ {bayes_thresh:,} SAR)"), use_container_width=True)
    st.dataframe(bayes_df, use_container_width=True, hide_index=True)
    st.info(f"P(B) = P(High Spender) = {p_hs:.4f}. Each bar = posterior probability of being from that country given the pilgrim is a high spender.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Poisson Distribution (Week 10) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Poisson Distribution — Group Size Modelling  |  Week 10</div>', unsafe_allow_html=True)
    lam = dff["Group_Size"].mean()
    k_vals = np.arange(0, int(dff["Group_Size"].max())+2)
    pmf_vals = stats.poisson.pmf(k_vals, lam)
    emp_freq = dff["Group_Size"].value_counts(normalize=True).reindex(k_vals, fill_value=0)

    fp = go.Figure()
    fp.add_trace(go.Bar(x=k_vals, y=emp_freq.values, name="Empirical (Observed)",
                        marker_color="#F4A96A", marker_line_width=0, opacity=0.75))
    fp.add_trace(go.Scatter(x=k_vals, y=pmf_vals, mode="lines+markers",
                            name=f"Poisson PMF (λ = {lam:.2f})",
                            line=dict(color="#D95F0A", width=2),
                            marker=dict(size=7, color="#D95F0A")))
    fp.update_layout(xaxis_title="Group Size (k)", yaxis_title="P(X = k)",
                     legend=dict(orientation="h"))
    st.plotly_chart(T(fp, f"Poisson PMF vs Empirical — λ = {lam:.2f}"), use_container_width=True)

    p1c, p2c, p3c = st.columns(3)
    p1c.metric("λ (Mean = Variance)", f"{lam:.4f}")
    p2c.metric("P(Group Size = 1)",   f"{stats.poisson.pmf(1, lam):.4f}")
    p3c.metric("P(Group Size ≤ 2)",   f"{stats.poisson.cdf(2, lam):.4f}")
    st.info(f"Poisson PMF: P(X = k) = e⁻λ · λᵏ / k!  where λ = {lam:.2f}. CDF: P(X ≤ k) = Σ P(X = i) for i = 0..k.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Binomial Distribution (Week 10) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Binomial Distribution — P(Male Pilgrim in Group)  |  Week 10</div>', unsafe_allow_html=True)
    p_male = (dff["Gender"] == "Male").mean()
    binom_n = st.slider("Group size n (number of trials)", 1, 20, 10, key="binom_n")
    binom_k_vals = np.arange(0, binom_n + 1)
    binom_pmf = stats.binom.pmf(binom_k_vals, binom_n, p_male)
    binom_cdf = stats.binom.cdf(binom_k_vals, binom_n, p_male)

    fb1, fb2 = st.columns(2)
    with fb1:
        fig_binom = go.Figure()
        fig_binom.add_trace(go.Bar(x=binom_k_vals, y=binom_pmf,
                                   marker_color="#D95F0A", marker_line_width=0,
                                   name="PMF"))
        fig_binom.update_layout(xaxis_title="k (number of males)", yaxis_title="P(X = k)")
        st.plotly_chart(T(fig_binom, f"Binomial PMF — n={binom_n}, p={p_male:.3f}"), use_container_width=True)
    with fb2:
        fig_bcdf = go.Figure()
        fig_bcdf.add_trace(go.Scatter(x=binom_k_vals, y=binom_cdf,
                                      mode="lines+markers", line=dict(color="#D95F0A", width=2),
                                      marker=dict(size=7), name="CDF"))
        fig_bcdf.update_layout(xaxis_title="k", yaxis_title="P(X ≤ k)")
        st.plotly_chart(T(fig_bcdf, f"Binomial CDF — n={binom_n}, p={p_male:.3f}"), use_container_width=True)

    bm1, bm2, bm3 = st.columns(3)
    bm1.metric("p (P(Male))",        f"{p_male:.4f}")
    bm2.metric("E(X) = np",          f"{binom_n * p_male:.4f}")
    bm3.metric("Var(X) = np(1-p)",   f"{binom_n * p_male * (1 - p_male):.4f}")
    st.info(f"Binomial formula: P(X = k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ  |  n = {binom_n}, p = {p_male:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hypergeometric Distribution (Week 10) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Hypergeometric Distribution — Sampling Without Replacement  |  Week 10</div>', unsafe_allow_html=True)
    st.caption("Population N, K successes in population, draw n samples. P(X=k) = C(K,k)·C(N-K,n-k) / C(N,n)")

    hg_col1, hg_col2, hg_col3 = st.columns(3)
    N_pop  = len(dff)
    K_suc  = int((dff["Gender"] == "Male").sum())
    hg_n   = hg_col1.slider("Sample size (n)", 5, 50, 20, key="hg_n")
    hg_k_vals = np.arange(0, min(hg_n, K_suc) + 1)
    hg_pmf = stats.hypergeom.pmf(hg_k_vals, N_pop, K_suc, hg_n)

    hg_col2.metric("Population N",      f"{N_pop:,}")
    hg_col3.metric("Males K in pop",    f"{K_suc:,}")

    fig_hg = go.Figure()
    fig_hg.add_trace(go.Bar(x=hg_k_vals, y=hg_pmf,
                             marker_color="#D95F0A", marker_line_width=0, name="PMF"))
    fig_hg.update_layout(xaxis_title="k (males in sample)", yaxis_title="P(X = k)")
    st.plotly_chart(T(fig_hg, f"Hypergeometric PMF — N={N_pop}, K={K_suc}, n={hg_n}"), use_container_width=True)

    hg_mean = hg_n * K_suc / N_pop
    hg_var  = hg_n * (K_suc/N_pop) * ((N_pop - K_suc)/N_pop) * ((N_pop - hg_n)/(N_pop - 1))
    hgm1, hgm2 = st.columns(2)
    hgm1.metric("E(X) = nK/N",  f"{hg_mean:.4f}")
    hgm2.metric("Var(X)",       f"{hg_var:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Normal Distribution (Week 11) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Normal Distribution — Estimated Spending  |  Week 11</div>', unsafe_allow_html=True)
    spend = dff["Estimated_Spending_SAR"].dropna()
    mu, sigma = spend.mean(), spend.std()
    x_r = np.linspace(spend.min(), spend.max(), 400)
    pdf_v = stats.norm.pdf(x_r, mu, sigma)

    fn = go.Figure()
    fn.add_trace(go.Histogram(x=spend, nbinsx=50, histnorm="probability density",
                              name="Observed Data", marker_color="#F4A96A",
                              marker_line_width=0, opacity=0.7))
    fn.add_trace(go.Scatter(x=x_r, y=pdf_v, mode="lines",
                            name=f"Normal Curve  μ={mu:,.0f}, σ={sigma:,.0f}",
                            line=dict(color="#D95F0A", width=2.5)))
    fn.update_layout(xaxis_title="Estimated Spending (SAR)", yaxis_title="Probability Density",
                     legend=dict(orientation="h"))
    st.plotly_chart(T(fn, "Histogram + Normal Curve — Estimated Spending (SAR)"), use_container_width=True)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Mean (μ)",              f"{mu:,.2f} SAR")
    n2.metric("Std Dev (σ)",           f"{sigma:,.2f} SAR")
    n3.metric("P(X < μ)",              "0.5000")
    n4.metric("P(μ−σ < X < μ+σ)",     "≈ 0.6827")
    st.info(f"Empirical rule: ~68% spend between {mu-sigma:,.0f} and {mu+sigma:,.0f} SAR · ~95% between {mu-2*sigma:,.0f} and {mu+2*sigma:,.0f} SAR")

    # Area under the curve (Week 11)
    st.markdown("**Area Under the Normal Curve — P(a < X < b)  |  Week 11**")
    az1, az2 = st.columns(2)
    with az1:
        a_val = st.number_input("Lower bound (a)", value=int(mu - sigma), step=500)
    with az2:
        b_val = st.number_input("Upper bound (b)", value=int(mu + sigma), step=500)

    prob_area = stats.norm.cdf(b_val, mu, sigma) - stats.norm.cdf(a_val, mu, sigma)
    x_fill = np.linspace(a_val, b_val, 200)
    y_fill = stats.norm.pdf(x_fill, mu, sigma)

    fn2 = go.Figure()
    fn2.add_trace(go.Scatter(x=x_r, y=pdf_v, mode="lines",
                             name="Normal Curve", line=dict(color="#D95F0A", width=2)))
    fn2.add_trace(go.Scatter(
        x=np.concatenate([[a_val], x_fill, [b_val]]),
        y=np.concatenate([[0], y_fill, [0]]),
        fill="toself", fillcolor="rgba(217,95,10,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"P({a_val:,} < X < {b_val:,}) = {prob_area:.4f}"
    ))
    fn2.update_layout(xaxis_title="Spending (SAR)", yaxis_title="Density",
                      legend=dict(orientation="h"))
    st.plotly_chart(T(fn2, "Area Under the Normal Curve"), use_container_width=True)
    st.success(f"P({a_val:,} < X < {b_val:,}) = {prob_area:.4f}  ({prob_area*100:.2f}% of pilgrims)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Standard Normal / Z-score (Week 11) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Standard Normal Distribution — Z-Score Calculator  |  Week 11</div>', unsafe_allow_html=True)
    z1, z2 = st.columns(2)
    with z1:
        x_val   = st.number_input("Enter spending value (X)", value=int(mu), step=500)
        z_score = (x_val - mu) / sigma
        p_less  = stats.norm.cdf(z_score)
        p_more  = 1 - p_less
        st.metric("Z-Score",       f"{z_score:.4f}")
        st.metric("P(X < value)",  f"{p_less:.4f}")
        st.metric("P(X > value)",  f"{p_more:.4f}")
    with z2:
        x_std = np.linspace(-4, 4, 300)
        y_std = stats.norm.pdf(x_std, 0, 1)
        fz = go.Figure()
        fz.add_trace(go.Scatter(x=x_std, y=y_std, mode="lines",
                                name="Standard Normal", line=dict(color="#D95F0A", width=2)))
        x_shade = np.linspace(-4, z_score, 200)
        y_shade = stats.norm.pdf(x_shade, 0, 1)
        fz.add_trace(go.Scatter(
            x=np.concatenate([[-4], x_shade, [z_score]]),
            y=np.concatenate([[0], y_shade, [0]]),
            fill="toself", fillcolor="rgba(217,95,10,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"P(Z < {z_score:.2f}) = {p_less:.4f}"
        ))
        fz.update_layout(xaxis_title="Z", yaxis_title="φ(Z)", legend=dict(orientation="h"))
        st.plotly_chart(T(fz, "Standard Normal Curve — Shaded Area"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — CONFIDENCE INTERVALS & HYPOTHESIS TESTING
# Week 12: Point estimation, interval estimation, CI for mean,
#           z-test and t-test for single mean
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Confidence Intervals":
    st.markdown('<span class="pg-title">Confidence Intervals & Hypothesis Testing</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Point estimation, interval estimation, z-test and t-test for single mean  |  Week 12</p>', unsafe_allow_html=True)

    # ── Point estimation (Week 12) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Point Estimation — Sample Statistics as Population Estimates  |  Week 12</div>', unsafe_allow_html=True)
    pt_rows = []
    for col, label in [("Estimated_Spending_SAR","Spending (SAR)"),
                        ("Group_Size","Group Size"),
                        ("Stay_Days","Stay Days")]:
        s = dff[col]
        pt_rows.append({
            "Variable":             label,
            "Point Estimate (x̄)":  round(s.mean(), 4),
            "Sample Std Dev (s)":   round(s.std(), 4),
            "Sample Size (n)":      len(s),
            "Std Error (s/√n)":     round(s.std() / np.sqrt(len(s)), 4),
        })
    st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)
    st.info("The sample mean x̄ is the best point estimate of the population mean μ. Standard error = s / √n measures the precision of the estimate.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Confidence interval (Week 12) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Confidence Interval for Mean — Spending by Country  |  Week 12</div>', unsafe_allow_html=True)
    conf_level = st.slider("Confidence Level (%)", 80, 99, 95) / 100
    alpha = 1 - conf_level

    ci_rows = []
    for country, grp in dff.groupby("Country")["Estimated_Spending_SAR"]:
        n_c  = len(grp)
        mean = grp.mean()
        se   = grp.std() / np.sqrt(n_c)
        if n_c >= 30:
            crit = stats.norm.ppf(1 - alpha/2)
            dist = "z"
        else:
            crit = stats.t.ppf(1 - alpha/2, df=n_c-1)
            dist = "t"
        ci_rows.append({
            "Country":             country,
            "n":                   n_c,
            "Mean (SAR)":          round(mean, 2),
            "Std Error":           round(se, 4),
            f"Critical ({dist})":  round(crit, 4),
            "CI Lower":            round(mean - crit*se, 2),
            "CI Upper":            round(mean + crit*se, 2),
            "Margin of Error":     round(crit*se, 2),
        })
    ci_df = pd.DataFrame(ci_rows).sort_values("Mean (SAR)", ascending=False)
    st.dataframe(ci_df, use_container_width=True, hide_index=True)

    # CI chart (simple bar with error bars — no forest plot label needed)
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
    ffp.update_layout(xaxis_title="Estimated Spending (SAR)", yaxis_title="Country")
    st.plotly_chart(T(ffp, f"{int(conf_level*100)}% Confidence Intervals for Mean Spending by Country"), use_container_width=True)
    st.info(f"A {int(conf_level*100)}% CI means: if we repeated this sampling process many times, {int(conf_level*100)}% of constructed intervals would contain the true population mean μ.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── z-test for single mean (Week 12) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">z-Test for Single Mean  |  Week 12</div>', unsafe_allow_html=True)
    st.caption("Use when population standard deviation σ is known, or n ≥ 30 (Central Limit Theorem applies).")

    hyp_mu  = st.number_input("H₀: Hypothesised Population Mean μ₀ (SAR)", value=22000, step=500, key="ztest_mu")
    alpha_z = st.selectbox("Significance Level α", [0.01, 0.05, 0.10], index=1, key="ztest_alpha")

    x_bar   = dff["Estimated_Spending_SAR"].mean()
    n_z     = len(dff)
    sigma_z = dff["Estimated_Spending_SAR"].std()
    se_z    = sigma_z / np.sqrt(n_z)
    z_stat  = (x_bar - hyp_mu) / se_z
    z_crit  = stats.norm.ppf(1 - alpha_z/2)
    p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    reject_z = abs(z_stat) > z_crit

    zc1, zc2, zc3, zc4 = st.columns(4)
    zc1.metric("Sample Mean (x̄)",  f"{x_bar:,.2f} SAR")
    zc2.metric("z-Statistic",       f"{z_stat:.4f}")
    zc3.metric("Critical Value ±",  f"{z_crit:.4f}")
    zc4.metric("p-Value",           f"{p_val_z:.4f}")

    if reject_z:
        st.error(f"**Reject H₀** — |z| = {abs(z_stat):.4f} > z_crit = {z_crit:.4f}. Sufficient evidence that μ ≠ {hyp_mu:,} SAR at α = {alpha_z}")
    else:
        st.success(f"**Fail to Reject H₀** — |z| = {abs(z_stat):.4f} ≤ z_crit = {z_crit:.4f}. Insufficient evidence to reject μ = {hyp_mu:,} SAR at α = {alpha_z}")

    x_z   = np.linspace(-4, 4, 300)
    y_z   = stats.norm.pdf(x_z, 0, 1)
    fztest = go.Figure()
    fztest.add_trace(go.Scatter(x=x_z, y=y_z, mode="lines",
                                line=dict(color="#888888", width=1.5), name="Standard Normal"))
    for x_rej, side in [(np.linspace(-4, -z_crit, 100), "left"),
                         (np.linspace(z_crit, 4, 100),  "right")]:
        fztest.add_trace(go.Scatter(
            x=np.concatenate([[x_rej[0]], x_rej, [x_rej[-1]]]),
            y=np.concatenate([[0], stats.norm.pdf(x_rej), [0]]),
            fill="toself", fillcolor="rgba(217,95,10,0.3)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Rejection Region" if side == "right" else "", showlegend=(side == "right")))
    fztest.add_vline(x=z_stat, line_color="#D95F0A", line_width=2.5,
                     annotation_text=f"z = {z_stat:.3f}", annotation_position="top")
    fztest.update_layout(xaxis_title="z", yaxis_title="φ(z)", legend=dict(orientation="h"))
    st.plotly_chart(T(fztest, "z-Test — Rejection Regions and Test Statistic"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── t-test for single mean (Week 12) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">t-Test for Single Mean  |  Week 12</div>', unsafe_allow_html=True)
    st.caption("Use when population standard deviation σ is unknown and estimated by sample std s.")

    hyp_mu_t = st.number_input("H₀: Hypothesised Population Mean μ₀ (SAR)", value=22000, step=500, key="ttest_mu")
    alpha_t  = st.selectbox("Significance Level α", [0.01, 0.05, 0.10], index=1, key="ttest_alpha")

    n_t    = len(dff)
    s_t    = dff["Estimated_Spending_SAR"].std()
    se_t   = s_t / np.sqrt(n_t)
    t_stat = (x_bar - hyp_mu_t) / se_t
    df_t   = n_t - 1
    t_crit = stats.t.ppf(1 - alpha_t/2, df=df_t)
    p_val_t = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))
    reject_t = abs(t_stat) > t_crit

    ci_lo_t = x_bar - t_crit * se_t
    ci_hi_t = x_bar + t_crit * se_t

    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("t-Statistic",        f"{t_stat:.4f}")
    tc2.metric("Degrees of Freedom", f"{df_t}")
    tc3.metric("Critical Value ±",   f"{t_crit:.4f}")
    tc4.metric("p-Value",            f"{p_val_t:.4f}")

    st.info(f"{int((1-alpha_t)*100)}% Confidence Interval: [{ci_lo_t:,.2f}, {ci_hi_t:,.2f}] SAR")

    if reject_t:
        st.error(f"**Reject H₀** — |t| = {abs(t_stat):.4f} > t_crit = {t_crit:.4f}. Sufficient evidence that μ ≠ {hyp_mu_t:,} SAR at α = {alpha_t}")
    else:
        st.success(f"**Fail to Reject H₀** — |t| = {abs(t_stat):.4f} ≤ t_crit = {t_crit:.4f}. Insufficient evidence to reject μ = {hyp_mu_t:,} SAR")

    x_t_range = np.linspace(-5, 5, 300)
    y_t_range = stats.t.pdf(x_t_range, df=df_t)
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=x_t_range, y=y_t_range, mode="lines",
                            line=dict(color="#888888", width=1.5), name=f"t-Distribution (df={df_t})"))
    for sign in [-1, 1]:
        x_rej = np.linspace(sign * t_crit, sign * 5, 100) if sign == 1 else np.linspace(-5, sign * t_crit, 100)
        ft.add_trace(go.Scatter(
            x=np.concatenate([[x_rej[0]], x_rej, [x_rej[-1]]]),
            y=np.concatenate([[0], stats.t.pdf(x_rej, df=df_t), [0]]),
            fill="toself", fillcolor="rgba(217,95,10,0.3)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Rejection Region" if sign == 1 else "", showlegend=(sign == 1)))
    ft.add_vline(x=t_stat, line_color="#D95F0A", line_width=2.5,
                 annotation_text=f"t = {t_stat:.3f}", annotation_position="top")
    ft.update_layout(xaxis_title="t", yaxis_title="f(t)", legend=dict(orientation="h"))
    st.plotly_chart(T(ft, f"t-Test — Rejection Regions  (df = {df_t})"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — REGRESSION & PREDICTIONS
# Week 13: SLR, correlation, testing of correlation coefficient,
#           coefficient of determination (R²)
# Week 14: Testing of slope, ANOVA for overall regression significance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Regression & Predictions":
    st.markdown('<span class="pg-title">Regression & Predictions</span>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">SLR, correlation, testing of correlation coefficient, R², slope test, ANOVA  |  Weeks 13–14</p>', unsafe_allow_html=True)

    dfe = dff.copy()
    # Only dataset variables (no derived spending_per_day etc.)
    num_cols = ["Estimated_Spending_SAR", "Group_Size", "Stay_Days"]

    # ── Pearson Correlation (Week 13) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Pearson Correlation Matrix  |  Week 13</div>', unsafe_allow_html=True)
    corr_matrix = dfe[num_cols].corr().round(4)
    fco = px.imshow(corr_matrix,
                    color_continuous_scale=[[0,"#FDE0C5"],[0.5,"#F4A96A"],[1,"#A03A04"]],
                    text_auto=".3f", zmin=-1, zmax=1, aspect="auto",
                    labels={"color":"r"})
    st.plotly_chart(T(fco, "Correlation Matrix — Pearson r"), use_container_width=True)
    st.info("Pearson r: |r| > 0.7 = strong, 0.4–0.7 = moderate, < 0.4 = weak.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Testing of Correlation Coefficient (Week 13) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Testing of Correlation Coefficient  |  Week 13</div>', unsafe_allow_html=True)
    st.caption("H₀: ρ = 0 (no linear correlation)  vs  H₁: ρ ≠ 0   |   Test statistic: t = r√(n−2) / √(1−r²)")

    tc_x = st.selectbox("Variable X", num_cols, index=1, key="tc_x")
    tc_y = st.selectbox("Variable Y", num_cols, index=0, key="tc_y")
    alpha_r = st.selectbox("Significance Level α", [0.01, 0.05, 0.10], index=1, key="rc_alpha")

    r_val, p_r = stats.pearsonr(dfe[tc_x], dfe[tc_y])
    n_r  = len(dfe)
    t_r  = r_val * np.sqrt(n_r - 2) / np.sqrt(1 - r_val**2)
    tc_crit = stats.t.ppf(1 - alpha_r/2, df=n_r - 2)
    reject_r = abs(t_r) > tc_crit

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Pearson r",        f"{r_val:.4f}")
    rc2.metric("t-Statistic",      f"{t_r:.4f}")
    rc3.metric("Critical Value ±", f"{tc_crit:.4f}")
    rc4.metric("p-Value",          f"{p_r:.4f}")

    if reject_r:
        st.error(f"**Reject H₀** — significant linear correlation between {tc_x.replace('_',' ')} and {tc_y.replace('_',' ')} at α = {alpha_r} (|t| = {abs(t_r):.4f} > {tc_crit:.4f})")
    else:
        st.success(f"**Fail to Reject H₀** — no significant linear correlation at α = {alpha_r}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── SLR (Week 13) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Simple Linear Regression (SLR)  |  Week 13</div>', unsafe_allow_html=True)
    slr_c1, slr_c2 = st.columns(2)
    with slr_c1:
        x_var = st.selectbox("Independent Variable (X)", ["Stay_Days","Group_Size"], index=0)
    with slr_c2:
        y_var = st.selectbox("Dependent Variable (Y)",   ["Estimated_Spending_SAR"], index=0)

    X_slr   = sm.add_constant(dfe[x_var])
    mdl_slr = sm.OLS(dfe[y_var], X_slr).fit()
    b0      = mdl_slr.params["const"]
    b1      = mdl_slr.params[x_var]
    r2      = mdl_slr.rsquared
    r_slr   = np.sqrt(r2) * np.sign(b1)
    se_b1   = mdl_slr.bse[x_var]
    t_b1    = mdl_slr.tvalues[x_var]
    p_b1    = mdl_slr.pvalues[x_var]
    df_res  = mdl_slr.df_resid
    t_crit_slr = stats.t.ppf(0.975, df=df_res)

    fsl = px.scatter(dfe, x=x_var, y=y_var,
                     color="Accommodation_Type", color_discrete_sequence=ORANGE_SEQ,
                     opacity=0.45,
                     labels={x_var: x_var.replace("_"," "), y_var: y_var.replace("_"," ")})
    xl = np.array([dfe[x_var].min(), dfe[x_var].max()])
    fsl.add_trace(go.Scatter(x=xl, y=b0 + b1*xl, mode="lines", name="Regression Line",
                             line=dict(color="#111111", width=2, dash="dash")))
    fsl.update_layout(legend_title="Accommodation")
    st.plotly_chart(T(fsl, f"Scatter + Regression Line: ŷ = {b0:.2f} + {b1:.2f} · {x_var.replace('_',' ')}"), use_container_width=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Intercept (β₀)",              f"{b0:,.4f}")
    m2.metric("Slope (β₁)",                  f"{b1:,.4f}")
    m3.metric("r (Correlation)",             f"{r_slr:.4f}")
    m4.metric("R² (Coeff. of Determination)", f"{r2:.4f}")
    m5.metric("Std Error of β₁",             f"{se_b1:.4f}")

    st.info(f"**Equation:** ŷ = {b0:,.2f} + {b1:,.2f}·x  |  R² = {r2:.4f} → {r2*100:.2f}% of variance in {y_var.replace('_',' ')} explained by {x_var.replace('_',' ')}.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Testing of slope (Week 14) ──
    st.markdown('<div class="chart-wrap"><div class="chart-title">Testing of Slope (β₁)  |  Week 14</div>', unsafe_allow_html=True)
    st.caption("H₀: β₁ = 0  vs  H₁: β₁ ≠ 0")

    alpha_slr    = st.selectbox("Significance Level α", [0.01, 0.05, 0.10], index=1, key="slr_alpha")
    t_crit_slope = stats.t.ppf(1 - alpha_slr/2, df=df_res)
    reject_slope = abs(t_b1) > t_crit_slope

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("t-Statistic (β₁)", f"{t_b1:.4f}")
    sc2.metric("Critical Value ±", f"{t_crit_slope:.4f}")
    sc3.metric("p-Value",          f"{p_b1:.4f}")
    sc4.metric("df (residual)",    f"{int(df_res)}")

    if reject_slope:
        st.error(f"**Reject H₀** — slope β₁ is statistically significant at α = {alpha_slr}. {x_var.replace('_',' ')} significantly predicts {y_var.replace('_',' ')}.")
    else:
        st.success(f"**Fail to Reject H₀** — slope β₁ NOT significant at α = {alpha_slr}.")
    st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown("---")

    # # ── ANOVA for overall regression (Week 14) ──
    # st.markdown('<div class="chart-wrap"><div class="chart-title">ANOVA Table — Overall Regression Significance  |  Week 14</div>', unsafe_allow_html=True)
    # st.caption("H₀: β₁ = 0 (model explains no variance)")

    # anova_table = sm.stats.anova_lm(mdl_slr, typ=1)
    # ss_reg  = anova_table["sum_sq"].iloc[0]
    # ss_res  = anova_table["sum_sq"].iloc[1]
    # ss_tot  = ss_reg + ss_res
    # df_reg  = int(anova_table["df"].iloc[0])
    # df_res_ = int(anova_table["df"].iloc[1])
    # ms_reg  = ss_reg / df_reg
    # ms_res  = ss_res / df_res_
    # f_stat  = ms_reg / ms_res
    # p_f     = mdl_slr.f_pvalue

    # anova_df = pd.DataFrame({
    #     "Source":      ["Regression", "Residual (Error)", "Total"],
    #     "SS":          [round(ss_reg,2),  round(ss_res,2),  round(ss_tot,2)],
    #     "df":          [df_reg,           df_res_,           df_reg + df_res_],
    #     "MS":          [round(ms_reg,2),  round(ms_res,2),  "—"],
    #     "F-Statistic": [round(f_stat,4),  "—",              "—"],
    #     "p-Value":     [round(p_f,4),     "—",              "—"],
    # })
    # st.dataframe(anova_df, use_container_width=True, hide_index=True)

    # fa1, fa2, fa3 = st.columns(3)
    # fa1.metric("F-Statistic", f"{f_stat:.4f}")
    # fa2.metric("p-Value",     f"{p_f:.4f}")
    # fa3.metric("Decision",    "Reject H₀" if p_f < 0.05 else "Fail to Reject H₀")

    # if p_f < 0.05:
    #     st.error(f"**Reject H₀** — overall regression is significant (F = {f_stat:.4f}, p = {p_f:.4f}).")
    # else:
    #     st.success(f"**Fail to Reject H₀** — overall regression NOT significant (F = {f_stat:.4f}, p = {p_f:.4f}).")
    # st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown("---")

    # # ── Prediction (Week 13) ──
    # st.markdown('<div class="chart-wrap"><div class="chart-title">Prediction Using the SLR Model  |  Week 13</div>', unsafe_allow_html=True)
    # x_input = st.number_input(f"Enter {x_var.replace('_',' ')} value",
    #                            min_value=float(dfe[x_var].min()),
    #                            max_value=float(dfe[x_var].max()),
    #                            value=float(dfe[x_var].mean()))
    # y_pred  = b0 + b1 * x_input
    # pred_se = np.sqrt(mdl_slr.mse_resid * (1 + 1/len(dfe) + (x_input - dfe[x_var].mean())**2 / ((len(dfe)-1)*dfe[x_var].var())))
    # pi_lo   = y_pred - t_crit_slr * pred_se
    # pi_hi   = y_pred + t_crit_slr * pred_se

    # pr1, pr2, pr3 = st.columns(3)
    # pr1.metric("Predicted Value (ŷ)", f"{y_pred:,.2f} SAR")
    # pr2.metric("95% PI Lower",        f"{pi_lo:,.2f} SAR")
    # pr3.metric("95% PI Upper",        f"{pi_hi:,.2f} SAR")
    # st.info(f"ŷ = {b0:,.2f} + {b1:,.4f} × {x_input:.1f} = **{y_pred:,.2f} SAR**")
    # st.markdown('</div>', unsafe_allow_html=True)


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
        <div class="stat-item"><div class="s-label">Numeric</div><div class="s-val">3</div><div class="s-note">incl. Stay_Days</div></div>
        <div class="stat-item"><div class="s-label">Categorical</div><div class="s-val">6</div><div class="s-note">original</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Column Descriptions"):
        st.dataframe(pd.DataFrame({
            "Column": ["Pilgrim_ID","Country","Gender","Age_Group","Accommodation_Type",
                       "Transport_Type","Stay_Duration","Estimated_Spending_SAR","Group_Size","Stay_Days"],
            "Type":   ["ID","Categorical","Categorical","Categorical","Categorical",
                       "Categorical","Categorical","Numerical","Numerical","Derived"],
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
                "Stay_Duration parsed as integer days",
            ]
        }), use_container_width=True, hide_index=True)

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
        fq1 = px.histogram(dff, x="Estimated_Spending_SAR", nbins=30,
                           color_discrete_sequence=["#D95F0A"],
                           labels={"Estimated_Spending_SAR":"Spending (SAR)"})
        fq1.update_traces(marker_line_width=0)
        chart(fq1, "Histogram — Spending (SAR)")
    with qc2:
        fq2 = px.histogram(dff, x="Group_Size", nbins=5,
                           color_discrete_sequence=["#E8883A"],
                           labels={"Group_Size":"Group Size"})
        fq2.update_traces(marker_line_width=0)
        chart(fq2, "Histogram — Group Size")
    with qc3:
        fq3 = px.histogram(dff, x="Stay_Days", nbins=4,
                           color_discrete_sequence=["#F4A96A"],
                           labels={"Stay_Days":"Stay Duration (days)"})
        fq3.update_traces(marker_line_width=0)
        chart(fq3, "Histogram — Stay Duration (days)")