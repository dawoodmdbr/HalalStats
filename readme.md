import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hajj Pilgrim Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design Tokens ──────────────────────────────────────────────────────────
ACCENT   = "#F97316"
ACCENT_L = "#FFF7ED"
BORDER   = "#E5E7EB"
TEXT     = "#111827"
SUBTEXT  = "#6B7280"
BG       = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {TEXT}; background: {BG};
}}
section[data-testid="stSidebar"] {{
    background: #FAFAFA;
    border-right: 1px solid {BORDER};
}}
div[data-testid="metric-container"] {{
    background: #FAFAFA;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1rem 1.25rem;
}}
div[data-testid="metric-container"] > label {{
    font-size: 0.75rem !important;
    color: {SUBTEXT} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
div[data-testid="metric-container"] > div {{
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: {TEXT} !important;
}}
.pill {{
    display: inline-block;
    background: {ACCENT_L};
    color: {ACCENT};
    border: 1px solid #FED7AA;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 10px;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}}
.sec-title {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 0.2rem; }}
.week-tag {{
    font-size: 0.72rem; color: {SUBTEXT};
    font-style: italic; margin-bottom: 0.9rem; display: block;
}}
.sec-desc {{
    font-size: 0.84rem; color: {SUBTEXT};
    margin: 0 0 1.2rem; line-height: 1.6;
}}
hr {{ border: none; border-top: 1px solid {BORDER}; margin: 1.4rem 0; }}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────
def pill(label):
    st.markdown(f'<div class="pill">{label}</div>', unsafe_allow_html=True)

def section(title, desc, week=""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if week:
        st.markdown(f'<span class="week-tag">📅 Course Outline — {week}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)

# ─── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load(path):
    df = pd.read_csv(path)
    if "Stay_Duration" in df.columns:
        df["Stay_Duration_Days"] = df["Stay_Duration"].str.extract(r"(\d+)").astype(float)
    return df

CSV_PATH = "synthetic_hajj_dataset.csv"
if not os.path.exists(CSV_PATH):
    st.error(f"Dataset `{CSV_PATH}` not found. Place it in the same folder as app.py.")
    st.stop()

df = load(CSV_PATH)

NUMERIC_COLS = ["Estimated_Spending_SAR", "Group_Size", "Stay_Duration_Days"]
CAT_COLS     = ["Country", "Gender", "Age_Group", "Accommodation_Type",
                "Transport_Type", "Stay_Duration"]

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Hajj Pilgrim\nStatistical Analyzer")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size:0.78rem;color:{SUBTEXT}'>"
        f"Dataset: `{CSV_PATH}`<br>{len(df):,} pilgrims · {df.shape[1]} variables</span>",
        unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Overview",
        "📋  Raw Data",
        "📊  Bar Chart",
        "🥧  Pie Chart",
        "📐  Measures of Central Tendency",
        "📏  Measures of Dispersion",
        "📉  Frequency Distribution",
        "📈  Histogram",
        "📦  Box Plot",
        "🌡️  Correlation & Heatmap",
    ])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size:0.7rem;color:{SUBTEXT}'>"
        f"<b>Course:</b> MT2005 Probability & Statistics<br>"
        f"<b>University:</b> NUCES FAST<br>"
        f"<b>Scope:</b> Weeks 1–3 (CLO 1)</span>",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    pill("Dataset Overview")
    st.title("Hajj Pilgrim Statistical Analyzer")
    st.markdown(
        "<div class='sec-desc'>Exploring the synthetic Hajj pilgrim dataset using descriptive "
        "statistics and graphical techniques — aligned with MT2005 Course Outline, Weeks 1–3 (CLO 1).</div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pilgrims",        f"{len(df):,}")
    c2.metric("Variables",             f"{df.shape[1]}")
    c3.metric("Countries Represented", f"{df['Country'].nunique()}")
    c4.metric("Missing Values",        int(df.isnull().sum().sum()))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Variable Classification")

    var_info = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": [
            "Quantitative – Discrete" if c in ["Pilgrim_ID", "Group_Size"]
            else "Quantitative – Continuous" if c in ["Estimated_Spending_SAR", "Stay_Duration_Days"]
            else "Qualitative" for c in df.columns
        ],
        "Measurement Scale": [
            "Ratio" if c in ["Pilgrim_ID", "Group_Size", "Estimated_Spending_SAR", "Stay_Duration_Days"]
            else "Nominal" for c in df.columns
        ],
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Example":       [str(df[c].iloc[0]) for c in df.columns],
    })
    st.dataframe(var_info, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.info(
        "**Scope:** This app covers **Weeks 1–3** of MT2005 — data types, measurement scales, "
        "frequency distributions, graphical representation (bar chart, pie chart, histogram, dot plot), "
        "and measures of central tendency and dispersion for ungrouped and grouped data.")


# ════════════════════════════════════════════════════════════════════════════
# RAW DATA
# ════════════════════════════════════════════════════════════════════════════
elif "Raw Data" in page:
    pill("Exploration")
    section("Raw Dataset",
            "Unprocessed data table — each row is one pilgrim record. "
            "Understanding the raw data (sample, population, variables) is the first step before any analysis.",
            "Week 1 — Introduction, types of variables, sample vs population")
    n = st.slider("Rows to display", 5, 200, 20)
    st.dataframe(
        df.drop(columns=["Stay_Duration_Days"], errors="ignore").head(n),
        use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# BAR CHART
# ════════════════════════════════════════════════════════════════════════════
elif "Bar Chart" in page:
    pill("Graphical Representation")
    section("Bar Chart",
            "A bar chart displays the frequency of each category in a qualitative variable. "
            "Each bar's height equals the count of observations in that group. "
            "Bars are separated to emphasise the discrete nature of categories.",
            "Week 1 — Graphical representation: Bar chart")

    col = st.selectbox("Categorical variable", CAT_COLS)
    vc  = df[col].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(vc.index.astype(str), vc.values,
                  color=ACCENT, alpha=0.85, edgecolor=BG, linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vc.values) * 0.01,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=8, color=SUBTEXT)
    ax.set_xlabel(col, labelpad=8)
    ax.set_ylabel("Frequency (Count)")
    ax.set_title(f"Bar Chart — Frequency of '{col}'", fontweight="bold", pad=10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        f"**Interpretation:** The most frequent category is **{vc.index[0]}** "
        f"with **{vc.values[0]:,}** pilgrims ({vc.values[0]/len(df)*100:.1f}% of total).")


# ════════════════════════════════════════════════════════════════════════════
# PIE CHART
# ════════════════════════════════════════════════════════════════════════════
elif "Pie Chart" in page:
    pill("Graphical Representation")
    section("Pie Chart",
            "A pie chart shows each category's proportion relative to the whole (100%). "
            "Each slice angle = (frequency ÷ total) × 360°. "
            "Best used for qualitative data with a small number of categories (≤ 7).",
            "Week 1 — Graphical representation: Pie chart")

    col = st.selectbox("Categorical variable", CAT_COLS)
    top = st.slider("Top N categories (remainder → 'Other')", 3, 10, 6)

    vc = df[col].value_counts()
    if len(vc) > top:
        main  = vc.head(top)
        other = pd.Series({"Other": vc.iloc[top:].sum()})
        vc    = pd.concat([main, other])

    palette = ["#F97316","#FB923C","#FDBA74","#FED7AA",
               "#FEF3C7","#FFF7ED","#E5E7EB","#D1D5DB"][:len(vc)]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        vc.values, labels=vc.index, autopct="%1.1f%%",
        colors=palette, startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor=BG, linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(8); at.set_color(TEXT)
    ax.set_title(f"Pie Chart — Proportion of '{col}'", fontweight="bold", pad=12)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        f"**Interpretation:** **{vc.index[0]}** accounts for the largest share "
        f"at **{vc.values[0]/vc.sum()*100:.1f}%** of all pilgrims.")


# ════════════════════════════════════════════════════════════════════════════
# MEASURES OF CENTRAL TENDENCY
# ════════════════════════════════════════════════════════════════════════════
elif "Central Tendency" in page:
    pill("Descriptive Statistics")
    section("Measures of Central Tendency",
            "Central tendency locates the centre of a dataset. "
            "Mean = arithmetic average. Median = middle value when sorted. "
            "Mode = most frequent value. Trimmed Mean = mean after removing extreme top/bottom values.",
            "Week 2 — Mean, Median, Mode, Trimmed Mean (Ungrouped data)")

    col = st.selectbox("Numeric variable", NUMERIC_COLS)
    s   = df[col].dropna()

    mean_val   = s.mean()
    median_val = s.median()
    mode_vals  = s.mode()
    mode_val   = mode_vals.iloc[0] if not mode_vals.empty else float("nan")
    trim_n     = int(len(s) * 0.05)
    trimmed    = s.sort_values().iloc[trim_n: len(s) - trim_n].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean (x̄)",          f"{mean_val:,.2f}")
    c2.metric("Median",             f"{median_val:,.2f}")
    c3.metric("Mode",               f"{mode_val:,.2f}")
    c4.metric("Trimmed Mean (5%)",  f"{trimmed:,.2f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    if mean_val > median_val * 1.05:
        skew_note = "Mean > Median → distribution is likely **right-skewed** (a few very high values pull the mean up)."
    elif mean_val < median_val * 0.95:
        skew_note = "Mean < Median → distribution is likely **left-skewed**."
    else:
        skew_note = "Mean ≈ Median → distribution is approximately **symmetric**."

    st.markdown(f"""
**Definitions:**
- **Mean (x̄):** Sum of all values ÷ count. Formula: x̄ = Σxᵢ / n. Sensitive to outliers.
- **Median:** Middle value after sorting. If n is even, average of the two middle values. Robust to outliers.
- **Mode:** Value that appears most often. A dataset can have no mode, one mode, or multiple modes.
- **Trimmed Mean:** Remove the bottom and top 5% of values, then compute the mean — reduces outlier influence.

📌 {skew_note}
""")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(s, bins=25, color=ACCENT, alpha=0.75, edgecolor=BG)
    ax.axvline(mean_val,   color="#1D4ED8", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean_val:,.1f}")
    ax.axvline(median_val, color="#15803D", linewidth=1.8, linestyle=":",
               label=f"Median = {median_val:,.1f}")
    ax.axvline(mode_val,   color="#DC2626", linewidth=1.5, linestyle="-.",
               label=f"Mode = {mode_val:,.1f}")
    ax.set_xlabel(col); ax.set_ylabel("Frequency")
    ax.set_title(f"Mean, Median & Mode on Distribution of '{col}'", fontweight="bold", pad=10)
    ax.legend(fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MEASURES OF DISPERSION
# ════════════════════════════════════════════════════════════════════════════
elif "Dispersion" in page:
    pill("Descriptive Statistics")
    section("Measures of Dispersion",
            "Dispersion (spread) tells how much values differ from each other and from the center. "
            "Key measures: Range, Variance, Standard Deviation, Coefficient of Variation, IQR.",
            "Week 2 — Variance, Std Dev, CV, IQ Range, Range (Ungrouped data)")

    col = st.selectbox("Numeric variable", NUMERIC_COLS)
    s   = df[col].dropna()

    rng = s.max() - s.min()
    var = s.var()
    std = s.std()
    cv  = (std / s.mean()) * 100
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    iqr = q3 - q1

    c1, c2, c3 = st.columns(3)
    c1.metric("Range",               f"{rng:,.2f}")
    c1.metric("Variance (s²)",       f"{var:,.2f}")
    c2.metric("Std Deviation (s)",   f"{std:,.2f}")
    c2.metric("Coeff. of Variation", f"{cv:.2f}%")
    c3.metric("Q1",                  f"{q1:,.2f}")
    c3.metric("Q3",                  f"{q3:,.2f}")
    st.metric("IQR (Q3 − Q1)",       f"{iqr:,.2f}")

    st.markdown("<hr>", unsafe_allow_html=True)
    cv_interp = ("high variability" if cv > 30
                 else "moderate variability" if cv > 15
                 else "low variability — values are clustered near the mean")
    st.markdown(f"""
**Definitions:**
- **Range:** Max − Min. Simplest measure of spread. Highly sensitive to outliers.
- **Variance (s²):** Average of the squared deviations from the mean: s² = Σ(xᵢ − x̄)² / (n−1). Units are squared.
- **Standard Deviation (s):** √Variance. Same unit as the data — easier to interpret.
- **Coefficient of Variation (CV):** (s / x̄) × 100. Unit-free — allows comparison of spread across different variables.
- **IQR:** Q3 − Q1. Spread of the middle 50% of the data. Resistant to outliers.

📌 CV = **{cv:.1f}%** indicates **{cv_interp}** in `{col}`.
""")


# ════════════════════════════════════════════════════════════════════════════
# FREQUENCY DISTRIBUTION TABLE
# ════════════════════════════════════════════════════════════════════════════
elif "Frequency Distribution" in page:
    pill("Tabular Representation")
    section("Frequency Distribution Table",
            "Organises raw data into a table showing how often each value (or class interval) occurs. "
            "Columns: Frequency (f), Relative Frequency (RF), Percentage Frequency (PF), "
            "Cumulative Frequency (CF).",
            "Week 1 & 3 — Freq dist for qualitative & quantitative data, CF dist, RF, PF")

    mode = st.radio(
        "Variable type",
        ["Qualitative (Categorical)", "Quantitative (Numeric — grouped into class intervals)"],
        horizontal=True)

    if mode.startswith("Qualitative"):
        col  = st.selectbox("Categorical variable", CAT_COLS)
        vc   = df[col].value_counts().reset_index()
        vc.columns = ["Value", "Frequency (f)"]
        total = vc["Frequency (f)"].sum()
        vc["Relative Frequency (RF)"]   = (vc["Frequency (f)"] / total).round(4)
        vc["Percentage Freq (PF) %"]    = (vc["Relative Frequency (RF)"] * 100).round(2)
        vc["Cumulative Frequency (CF)"] = vc["Frequency (f)"].cumsum()
        st.dataframe(vc, use_container_width=True, hide_index=True)
        st.markdown(f"**n = {total:,}**")

    else:
        col  = st.selectbox("Numeric variable", NUMERIC_COLS)
        bins = st.slider("Number of class intervals", 4, 15, 7)
        s    = df[col].dropna()
        counts, edges = np.histogram(s, bins=bins)
        labels = [f"{edges[i]:,.1f} – {edges[i+1]:,.1f}" for i in range(len(counts))]
        total  = counts.sum()
        ft = pd.DataFrame({
            "Class Interval":           labels,
            "Frequency (f)":            counts,
            "Relative Frequency (RF)":  (counts / total).round(4),
            "Percentage Freq (PF) %":   ((counts / total) * 100).round(2),
            "Cumulative Frequency (CF)": np.cumsum(counts),
        })
        st.dataframe(ft, use_container_width=True, hide_index=True)
        st.markdown(
            f"**n = {total:,}** | Class width ≈ **{(s.max()-s.min())/bins:,.1f}**")

    st.markdown("""
**Key terms:**
- **f** — count of observations in that row / class
- **RF** = f ÷ n — proportion of total (decimal)
- **PF%** = RF × 100 — percentage
- **CF** — running total of frequencies (top to current row)
""")


# ════════════════════════════════════════════════════════════════════════════
# HISTOGRAM
# ════════════════════════════════════════════════════════════════════════════
elif "Histogram" in page:
    pill("Graphical Representation")
    section("Histogram",
            "A histogram is the graphical version of a frequency distribution for quantitative data. "
            "Unlike a bar chart, bars touch each other because data is continuous. "
            "The x-axis shows class intervals; y-axis shows frequency. "
            "Shape reveals skewness and modality.",
            "Week 3 — Graphical representation: Histogram")

    col  = st.selectbox("Numeric variable", NUMERIC_COLS)
    bins = st.slider("Number of bins (class intervals)", 5, 40, 20)
    s    = df[col].dropna()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(s, bins=bins, color=ACCENT, alpha=0.85, edgecolor=BG, linewidth=0.4)
    ax.axvline(s.mean(),   color="#1D4ED8", linewidth=1.5, linestyle="--",
               label=f"Mean = {s.mean():,.1f}")
    ax.axvline(s.median(), color="#15803D", linewidth=1.5, linestyle=":",
               label=f"Median = {s.median():,.1f}")
    ax.set_xlabel(col, labelpad=8)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram — Distribution of '{col}'", fontweight="bold", pad=10)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)

    skew = s.skew()
    if skew > 0.5:
        shape = f"**Right-skewed (positive skew = {skew:.2f})** — long tail to the right; most pilgrims have lower values but a few have very high values."
    elif skew < -0.5:
        shape = f"**Left-skewed (negative skew = {skew:.2f})** — long tail to the left."
    else:
        shape = f"**Approximately symmetric (skew = {skew:.2f})** — values are fairly evenly spread around the center."
    st.markdown(f"**Distribution shape:** {shape}")


# ════════════════════════════════════════════════════════════════════════════
# BOX PLOT
# ════════════════════════════════════════════════════════════════════════════
elif "Box Plot" in page:
    pill("Graphical Representation")
    section("Box Plot (Box-and-Whisker Plot)",
            "Visualises the five-point summary: Minimum, Q1 (25th percentile), Median, "
            "Q3 (75th percentile), Maximum. The box spans the IQR (middle 50% of data). "
            "Points beyond the whiskers (Q1 − 1.5·IQR  or  Q3 + 1.5·IQR) are potential outliers.",
            "Week 2 — Five-point summary, Box plot, IQR")

    col = st.selectbox("Numeric variable", NUMERIC_COLS)
    grp = st.selectbox("Group by (optional — compare across categories)", ["None"] + CAT_COLS)
    s   = df[col].dropna()

    q1, med, q3 = s.quantile(0.25), s.median(), s.quantile(0.75)
    iqr = q3 - q1
    outliers = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Min",    f"{s.min():,.1f}")
    c2.metric("Q1",     f"{q1:,.1f}")
    c3.metric("Median", f"{med:,.1f}")
    c4.metric("Q3",     f"{q3:,.1f}")
    c5.metric("Max",    f"{s.max():,.1f}")

    st.markdown(
        f"**IQR = {iqr:,.1f}** &nbsp;|&nbsp; "
        f"Whisker bounds: [{q1-1.5*iqr:,.1f}, {q3+1.5*iqr:,.1f}] &nbsp;|&nbsp; "
        f"Potential outliers: **{len(outliers)}** values")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if grp == "None":
        data, labels = [s.values], [col]
    else:
        groups = df[grp].dropna().unique()
        data   = [df[df[grp] == g][col].dropna().values for g in groups]
        labels = [str(g) for g in groups]

    bp = ax.boxplot(data, patch_artist=True, labels=labels,
                    medianprops=dict(color=ACCENT, linewidth=2.5),
                    whiskerprops=dict(color=SUBTEXT, linewidth=1),
                    capprops=dict(color=SUBTEXT, linewidth=1),
                    flierprops=dict(marker="o", markerfacecolor=ACCENT,
                                   markersize=4, alpha=0.5, markeredgewidth=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT_L)
        patch.set_edgecolor(ACCENT)
        patch.set_linewidth(1.5)

    title = f"Box Plot — '{col}'" + (f"  grouped by '{grp}'" if grp != "None" else "")
    ax.set_ylabel(col)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# CORRELATION & HEATMAP
# ════════════════════════════════════════════════════════════════════════════
elif "Correlation" in page:
    pill("Correlation")
    section("Pearson Correlation Coefficient & Heatmap",
            "The Pearson Correlation Coefficient (r) measures the strength and direction of a linear "
            "relationship between two quantitative variables. "
            "r = +1 → perfect positive, r = −1 → perfect negative, r = 0 → no linear relationship. "
            "The heatmap shows all pairwise correlations simultaneously.",
            "Week 13 — Correlation, Testing of correlation coefficient (CLO 3 reference)")

    corr = df[NUMERIC_COLS].corr().round(3)

    c1, c2 = st.columns(2)
    col_x  = c1.selectbox("Variable X", NUMERIC_COLS, 0)
    col_y  = c2.selectbox("Variable Y", NUMERIC_COLS, 1)

    clean = df[[col_x, col_y]].dropna()
    r, p  = pearsonr(clean[col_x], clean[col_y])
    strength  = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
    direction = "positive" if r > 0 else "negative"

    st.metric(f"Pearson r  ({col_x} vs {col_y})",
              f"{r:.4f}", delta=f"{direction} {strength} linear relationship")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                center=0, linewidths=0.5, linecolor=BG,
                square=True, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Heatmap", fontweight="bold", pad=12)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=True)

    st.markdown("""
**Reading the heatmap:**
- **Dark orange (→ +1):** Strong positive linear relationship
- **Dark blue (→ −1):** Strong negative linear relationship
- **White (≈ 0):** Little or no linear relationship

**Formula:** r = Σ[(xᵢ − x̄)(yᵢ − ȳ)] / √[Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²]
""")