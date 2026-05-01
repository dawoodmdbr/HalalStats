# 🕌 HalalStats — Hajj Pilgrim Analytics Dashboard

> A web-based statistical analysis dashboard built with Python & Streamlit for a **Probability and Statistics** semester project (Spring 2026).

---

## 📌 Project Overview

This dashboard performs comprehensive statistical analysis on a synthetic dataset of **2,000 Hajj pilgrims** from 10 countries. It covers graphical data exploration, descriptive statistics, inferential testing, probability distributions, and regression modelling — all presented through an interactive web interface.

---

## 🗂️ Dataset

**File:** `synthetic_hajj_dataset.csv`

| Variable | Type | Description |
|---|---|---|
| `Pilgrim_ID` | ID | Unique identifier (1–2000) |
| `Country` | Categorical | Country of origin (10 countries) |
| `Gender` | Categorical | Male / Female |
| `Age_Group` | Categorical | 18-30, 31-45, 46-60, 60+ |
| `Accommodation_Type` | Categorical | Camp (Mina), Apartment, Hotel 3★, Hotel 5★ |
| `Transport_Type` | Categorical | Bus, Group Transport, Train (Haramain), Private Car |
| `Stay_Duration` | Categorical | 7, 10, 14, or 21 days |
| `Estimated_Spending_SAR` | Numerical | Total spending in Saudi Riyals |
| `Group_Size` | Numerical | Number of persons in travel group (1–5) |
| `Stay_Days` *(derived)* | Numerical | Stay_Duration parsed as integer |
| `Spending_Per_Day` *(derived)* | Numerical | Spending ÷ Stay_Days |
| `Spending_Per_Person` *(derived)* | Numerical | Spending ÷ Group_Size |

---

## 📊 Charts, Diagrams & Why We Used Each One

Every visualisation in this dashboard was chosen deliberately. Below is a full reference of every chart, the graph type used, and the statistical justification for that choice.

---

### 📊 Page 1 — Overview

---

#### 1. Pilgrim Count by Country — **Vertical Bar Chart**

A vertical bar chart displays one bar per country, where the height of the bar represents the number of pilgrims from that country. We chose a bar chart here because the x-axis variable (Country) is categorical and unordered, and we are comparing a single count value across multiple distinct groups. Bar charts are the standard choice for comparing frequencies across categories because the human eye is very effective at judging the length of bars side by side. A pie chart would become cluttered with 10 categories, and a line chart would imply a continuous trend which does not exist here.

---

#### 2. Gender Distribution — **Donut Chart (Pie Chart with hole)**

A donut chart is a variant of a pie chart that displays proportional data as segments of a ring. We used it here because gender has only two categories (Male and Female), making proportional representation ideal — a reader immediately sees that one group is roughly half or more of the total. The hole in the centre is purely aesthetic; it reduces visual clutter and makes the chart look less heavy compared to a filled pie. We would not use a bar chart here because two-category proportion data is most naturally understood as a part-of-a-whole relationship.

---

#### 3. Pilgrim Count by Age Group — **Vertical Bar Chart**

Age groups (18-30, 31-45, 46-60, 60+) are ordered categorical bins. A vertical bar chart was chosen because it makes the count in each age bracket easy to compare at a glance. The natural left-to-right ordering of the bars also reflects the progression of age, giving the chart an intuitive flow. A histogram would be inappropriate here because age groups are pre-defined discrete categories, not a continuous variable.

---

#### 4. Accommodation Type Breakdown — **Donut Chart (Pie Chart with hole)**

Accommodation type has four categories and the question of interest is: *what share of pilgrims chose each accommodation?* A donut chart answers this as a part-of-a-whole breakdown. With only four categories, the segments are large enough to be readable without labels overlapping. A bar chart could also work, but a donut chart communicates proportion more naturally when the goal is to show composition rather than absolute counts.

---

#### 5. Spending Distribution by Country — **Box Plot**

A box plot (also called a box-and-whisker plot) shows the median, interquartile range (IQR), and outliers of a numerical variable for each group. We used it here because we want to compare not just the average spending across countries but also the spread and skewness — a simple bar chart of means would hide the variability. The box shows the middle 50% of spending values, the whiskers extend to the min/max within 1.5×IQR, and dots beyond the whiskers are outliers. This is much more statistically informative than a bar chart of means.

---

#### 6. Transport Type by Country — **Stacked Bar Chart**

A stacked bar chart shows a bar for each country, divided into coloured segments representing transport type proportions. We chose a stacked bar chart here because we are showing two categorical variables simultaneously: country (the groups) and transport type (the sub-groups). Stacking allows us to see both the total pilgrim count per country and the composition of transport types within each country in a single chart. A grouped bar chart would also work but would require more horizontal space with 4 transport types × 10 countries = 40 bars.

---

### 📈 Page 2 — Descriptive Statistics

---

#### 7. Histogram with Box Marginal — **Histogram + Marginal Box Plot**

A histogram divides a continuous variable into equal-width bins and plots the frequency (count) of values in each bin as vertical bars. We used it here to visualise the shape of the spending distribution — whether it is bell-shaped, skewed, or bimodal. The marginal box plot added above the histogram shows the median and quartiles at the same time, so the reader gets both the detailed distributional shape (histogram) and the five-number summary (box plot) in one view. This combination is standard practice when exploring a new numerical variable for the first time.

---

#### 8. Violin Plot by Gender — **Violin Plot**

A violin plot is like a box plot but adds a mirrored kernel density estimate on each side, showing the full shape of the distribution. We used it here split by gender to compare not just the median spending between Male and Female pilgrims but also where the density of values is concentrated. A standard box plot would only show the quartile boundaries; a violin plot reveals whether the distribution is unimodal, bimodal, or has a long tail. This is especially useful for detecting subtle differences between two groups that summary statistics alone would miss.

---

#### 9. Mean Spending by Country — **Horizontal Bar Chart (sorted)**

A horizontal bar chart was used here (bars going left to right) sorted in descending order of mean spending. We chose this layout specifically because country names are long strings — displaying them on the x-axis of a vertical bar chart causes label overlap and rotation. Placing the labels on the y-axis of a horizontal bar chart keeps them fully readable. Sorting by value also makes it easy to immediately identify which country spends the most and least, which is the primary question this chart answers.

---

#### 10. Spending by Accommodation — Mean / Median / Std Dev — **Grouped Bar Chart**

A grouped bar chart places multiple bars side by side within each category. We used it here to show three statistics (mean, median, and standard deviation) for spending across accommodation types simultaneously. Grouping allows direct visual comparison of all three measures for the same accommodation type without needing three separate charts. The difference between mean and median within each group also reveals skewness, and the standard deviation bar shows how spread out spending is for that accommodation type.

---

#### 11. Mean Spending Heatmap — Country × Age Group — **Heatmap**

A heatmap encodes a numerical value (mean spending) as a colour in a two-dimensional grid, where rows are countries and columns are age groups. We used a heatmap here because we are visualising a cross-tabulation of two categorical variables against a continuous outcome. A heatmap is the most compact and readable way to display a 10×4 matrix of values — a grouped bar chart with 40 bars would be impossible to read, and a table alone would require the reader to scan each cell mentally. The colour gradient (light to dark orange) makes high- and low-spending combinations immediately obvious.

---

#### 12. Correlation Matrix — **Heatmap (Correlation Heatmap)**

A correlation heatmap is a square grid where each cell shows the Pearson correlation coefficient between two numerical variables, encoded as colour. We used it here to display pairwise correlations among all five numerical variables in one view. Values range from -1 (perfect negative correlation, shown in light) to +1 (perfect positive correlation, shown in dark orange). This chart is the standard tool for a first-pass multicollinearity check before building regression models, and it quickly reveals which variables move together.

---

### 🔍 Page 3 — Confidence Intervals & Hypothesis Testing

---

#### 13. CI Table — **Data Table**

A plain data table is used here to present the exact numerical values of confidence intervals (lower bound, mean, upper bound) for each country. We chose a table rather than a chart because the primary use of this output is precision — the reader needs to know the exact numbers to report in a statistical analysis. The forest plot below provides the visual summary; the table provides the precise figures.

---

#### 14. Forest Plot — **Horizontal Error Bar Chart**

A forest plot (also called a confidence interval plot) shows a dot representing the point estimate (mean spending) for each group, with horizontal lines extending to the lower and upper bounds of the confidence interval. It is the standard visualisation used in statistics and medical research to communicate uncertainty around estimates. We used it here instead of a bar chart because the focus is on the interval width and the overlap between countries — overlapping confidence intervals suggest no significant difference between groups, while non-overlapping intervals suggest a significant difference. A bar chart of means would not show this uncertainty.

---

#### 15. Spending by Gender Box Plot — **Box Plot**

A box plot comparing Male vs. Female spending is used alongside the two-sample t-test to provide a visual complement to the numerical test result. The box plot shows whether the medians visually differ, how large the IQRs are relative to the difference, and whether outliers are pulling the means apart from the medians. This is best practice in hypothesis testing: never report a p-value without also showing the underlying distribution of both groups.

---

#### 16. Spending by Age Group (ANOVA) — **Box Plot**

A box plot with one box per age group is the appropriate companion visualisation for a one-way ANOVA. ANOVA tests whether any group means differ significantly, but it does not tell you which groups differ or how large the differences are. The box plot fills this gap visually — the reader can see the median and spread of each age group and form a judgment about practical significance beyond just statistical significance.

---

### 🎲 Page 4 — Probability & Distributions

---

#### 17. Normal Distribution Fit — **Histogram with PDF Overlay (Line Chart)**

The histogram shows the empirical distribution of spending values from the data. Overlaid on top is a smooth line representing the theoretical Normal distribution PDF calculated from the sample mean and standard deviation. We used this combination to visually assess goodness-of-fit: if the histogram bars closely follow the curve, the data is approximately normal. This is a standard diagnostic plot used before applying any parametric statistical test that assumes normality.

---

#### 18. Q-Q Plot (Quantile-Quantile Plot) — **Scatter Plot with Reference Line**

A Q-Q plot places the theoretical quantiles of a normal distribution on the x-axis and the observed sample quantiles on the y-axis. If the data is perfectly normal, all points fall exactly on the 45-degree reference line. Deviations from the line reveal the nature of non-normality: an S-curve suggests heavy tails; points above the line at the ends suggest skewness. We used the Q-Q plot here because it is a more sensitive normality diagnostic than the histogram — subtle departures from normality that are invisible on a histogram are clearly visible on a Q-Q plot.

---

#### 19. Poisson Distribution Fit — **Bar Chart with Line Overlay**

The bar chart shows the empirical probability of each group size value observed in the data. The orange line overlaid on top shows the theoretical Poisson PMF (probability mass function) calculated using the sample mean as λ. We chose this combination because group size is a discrete count variable — the exact type of variable the Poisson distribution models. Overlaying the theoretical PMF on the empirical bars lets us visually assess whether the Poisson distribution is a good model for group sizes in the dataset.

---

#### 20. Conditional Probability Bar Chart — **Vertical Bar Chart**

A simple vertical bar chart is used to show P(High Spender | Accommodation Type) — the probability that a pilgrim is a high spender, conditioned on their accommodation choice. Each bar represents one accommodation type and its height is the conditional probability as a percentage. A bar chart is the correct choice here because we are comparing a single probability value across four distinct categories. It is easy to read which accommodation type is associated with the highest probability of high spending.

---

### 📉 Page 5 — Regression & Predictions

---

#### 21. Simple Linear Regression Scatter Plot — **Scatter Plot with Regression Line**

A scatter plot places each pilgrim as a dot with Stay Days on the x-axis and Spending on the y-axis. The dashed regression line shows the best-fit linear relationship estimated by OLS. We used a scatter plot here because the goal is to visualise the linear relationship between two continuous variables. The colour coding by accommodation type adds a third dimension at no extra cost. The regression line makes the trend immediately visible, and the spread of points around the line conveys the strength of the relationship (reflected in R²).

---

#### 22. MLR Coefficient Plot — **Horizontal Bar Chart with Error Bars**

A coefficient plot is a horizontal bar chart where each bar represents one regression coefficient, with error bars showing the standard error. Bars are coloured orange if the coefficient is statistically significant at α = 0.05, and grey if not. We used this chart because a table of coefficients is hard to read visually — the coefficient plot lets you immediately see the magnitude and direction of each predictor's effect, and the error bars show how precise each estimate is. Overlapping error bars crossing zero indicate non-significant predictors.

---

#### 23. Residuals vs. Fitted Values — **Scatter Plot**

This scatter plot places the model's predicted values (fitted values) on the x-axis and the residuals (actual minus predicted) on the y-axis, with a horizontal dashed line at y=0. It is one of the four standard regression diagnostic plots. We use it to check the assumption of homoscedasticity (constant variance of residuals). If residuals are randomly scattered around the zero line with no pattern, the assumption holds. A funnel shape (wider spread at larger fitted values) would indicate heteroscedasticity, which violates OLS assumptions.

---

#### 24. Residual Distribution — **Histogram**

A histogram of the residuals is used to check the normality of residuals assumption in linear regression. If residuals follow a roughly bell-shaped, symmetric distribution centred at zero, the normality assumption is satisfied. A skewed or multi-peaked residual histogram suggests a model misspecification. This chart complements the Q-Q plot as a second visual check of the same assumption, and is standard practice in any regression analysis.

---

#### 25. Spending Predictor Gauge — **Gauge Chart (Indicator)**

A gauge chart displays the predicted spending value as a needle on a semicircular dial, similar to a speedometer. The dial is divided into colour-coded zones from low spending (light) to high spending (dark orange). We used a gauge chart here specifically for the interactive predictor because it communicates a single predicted value in a way that is immediately intuitive for a live demo — the viewer can see at a glance whether the prediction is in the low, medium, or high range relative to the overall dataset. A bar chart or number alone would be less engaging for presentation purposes.

---

### 🗃️ Page 6 — Raw Data

---

#### 26. Distribution Snapshots — **Histogram × 3**

Three small histograms at the bottom of the Raw Data page show the distribution of Spending, Group Size, and Stay Duration. We used histograms here as a quick sanity check for anyone exploring the raw data — they provide an immediate visual sense of the spread and shape of the three most important numerical variables without needing to navigate to the Descriptive Statistics page.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/halalstats.git
cd halalstats

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
statsmodels>=0.14.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🗃️ Project Structure

```
halalstats/
│
├── app.py                      # Main Streamlit application
├── synthetic_hajj_dataset.csv  # Dataset (must be in same folder as app.py)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Click the **☰ hamburger menu** (top-left) to open the sidebar
3. Navigate between the 6 pages using the sidebar menu
4. Use the **Filters** section in the sidebar to filter by Country, Gender, or Age Group — all charts update live
5. On the **Regression & Predictions** page, use the predictor form to estimate spending for a custom pilgrim profile
6. On the **Raw Data** page, use the search box and column selector to explore the dataset, then download a filtered CSV

---

## 📐 Statistical Methods Used

| Method | Chart / Output | Page |
|---|---|---|
| Frequency count | Vertical Bar Chart | Overview |
| Proportional composition | Donut Chart | Overview |
| Group comparison (counts) | Stacked Bar Chart | Overview |
| Five-number summary + outliers | Box Plot | Overview, Confidence Intervals |
| Mean, median, std, IQR, CV, skewness, kurtosis | Summary Table | Descriptive Statistics |
| Distributional shape | Histogram + Marginal Box | Descriptive Statistics |
| Group distribution comparison | Violin Plot | Descriptive Statistics |
| Cross-tabulation of two categoricals | Heatmap | Descriptive Statistics |
| Pairwise linear association | Correlation Heatmap | Descriptive Statistics |
| Confidence intervals (t-distribution) | Forest Plot + Table | Confidence Intervals |
| One-sample t-test | Metrics + Info box | Confidence Intervals |
| Two-sample independent t-test | Metrics + Box Plot | Confidence Intervals |
| One-way ANOVA | Metrics + Box Plot | Confidence Intervals |
| Empirical probability | Data Table | Probability & Distributions |
| Normal distribution fitting | Histogram + PDF Line | Probability & Distributions |
| Kolmogorov-Smirnov test | Metric | Probability & Distributions |
| Shapiro-Wilk normality test | Metric | Probability & Distributions |
| Q-Q plot | Scatter + Reference Line | Probability & Distributions |
| Poisson distribution fitting | Bar + PMF Line | Probability & Distributions |
| Conditional probability | Vertical Bar Chart | Probability & Distributions |
| Simple linear regression (OLS) | Scatter + Regression Line | Regression & Predictions |
| Multiple linear regression (OLS) | Coefficient Plot + Table | Regression & Predictions |
| Residual diagnostics | Scatter + Histogram | Regression & Predictions |
| Prediction with interval | Gauge Chart | Regression & Predictions |

---

## 📋 Report Structure (Submission Format)

This project follows the default submission format:

1. **Problem Statement** — Analysing spending patterns and demographic trends of Hajj pilgrims
2. **Objective** — Apply statistical methods to uncover insights from pilgrim data
3. **Data Description** — Synthetic dataset, 2,000 records, 9 original variables
4. **Results** — Screenshots of all dashboard pages
5. **Codes** — `app.py` with inline comments, font size 9, line spacing 1
6. **Conclusion** — Summary of key statistical findings

---

## ⚠️ Notes

- The dataset (`synthetic_hajj_dataset.csv`) **must be in the same directory** as `app.py`
- The app runs fully offline — no internet connection required after installation
- Tested on Python 3.10 and 3.11

---

## 📄 License

This project was created for academic purposes as part of a university semester project.

---

*Spring 2026 · Probability and Statistics · Department of Computer Science*