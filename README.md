# 🔍 BookScope: Exploratory Data Analysis
### CodeAlpha · Data Analytics Internship · Task 2: EDA

An interactive, question-driven **Exploratory Data Analysis** of the **1,000 books** scraped from
books.toscrape.com in **Task 1**. The analysis is written in Python (`pandas`, `scipy`) and presented
in a **Streamlit dashboard** with 7 sections.

---

## 🎯 Task requirements → where they are covered

| CodeAlpha requirement | Where in the project |
|---|---|
| Ask meaningful questions before analysis | **❓ Questions** tab: 7 questions (Q1–Q7), written before any result |
| Explore data structure, variables & data types | **🧱 Structure** tab: column types, missing values, unique counts, numeric summary |
| Identify trends, patterns and anomalies | **📊 Distributions** + **🔗 Relationships** tabs: histograms, box plots, heatmaps, IQR outliers |
| Test hypotheses using statistics & visualization | **🧪 Hypothesis tests** tab: 6 tests (Spearman, Kruskal-Wallis, Mann-Whitney, χ², KS) |
| Detect data issues for further analysis | **⚠️ Data issues** tab: 6 problems with evidence, severity and a fix |

---

## 📁 Project structure

```
CodeAlpha_ProjectName - data analytics (task 2)/
├── app.py               # Streamlit dashboard (the interface)
├── eda.py               # ALL analysis logic: questions, cleaning, tests, issues
├── data/books.csv       # dataset from Task 1
├── reports/eda_report.md# text report (made by: python eda.py)
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

## ⚙️ Setup & run (Windows cmd)

```cmd
cd "C:\Users\ILMA FARAZ\OneDrive\Desktop\CodeAlpha_ProjectName - data analytics (task 2)"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py        :: opens the dashboard at http://localhost:8501
python eda.py               :: prints the text report and saves reports/eda_report.md
```
> PowerShell: activate with `.\venv\Scripts\Activate.ps1`

---

## ❓ Questions & answers

| # | Question | Answer (from the data) |
|---|---|---|
| Q1 | Is the dataset clean enough to analyse? | 1,000 rows, 0 missing values. But 22% of books have a placeholder category and `in_stock` is always True |
| Q2 | How are prices distributed? | £10.00 to £59.99, median £35.98 ≈ mean £35.07. No skew, no outliers, **flat/uniform** |
| Q3 | How are ratings spread? | Each star level holds 18–23% of books, so they are **evenly spread** |
| Q4 | Which categories dominate? | Nonfiction, Sequential Art, Fiction. The biggest label, "Default" (152 books), is a **placeholder** |
| Q5 | Do higher-rated books cost more? | **No** (Spearman ρ = 0.03, p = 0.36) |
| Q6 | Do some categories cost more? | **No** significant difference among main genres (p = 0.22) |
| Q7 | Are series books rated differently? | **No** (313 series books, p = 0.79) |

## 🧪 Hypothesis tests (α = 0.05)

| ID | Hypothesis | Test | p-value | Result |
|---|---|---|---|---|
| H1 | Higher-rated books cost more | Spearman correlation | 0.356 | Fail to reject H0 |
| H2 | Price differs between categories | Kruskal-Wallis | 0.221 | Fail to reject H0 |
| H3 | Series books rated differently | Mann-Whitney U | 0.793 | Fail to reject H0 |
| H4 | Ratings evenly spread | χ² goodness-of-fit | 0.215 | Fail to reject H0 (uniform) |
| H5 | Prices uniformly distributed | Kolmogorov-Smirnov | 0.704 | Fail to reject H0 (uniform) |
| H6 | Rating depends on category | χ² independence | 0.579 | Fail to reject H0 |

**Why non-parametric tests?** Prices are not normally distributed and ratings are ordinal (1–5),
so rank-based tests (Spearman, Kruskal-Wallis, Mann-Whitney) are more appropriate than t-tests or ANOVA.

## ⚠️ Data issues found

1. **Placeholder categories (High):** 219 books (22%) in "Default" / "Add a comment"
2. **Constant column (Medium):** `in_stock` is True for every book, so it carries no information
3. **Tiny categories (Medium):** 15 of 50 categories have fewer than 5 books, so their averages are unreliable
4. **Duplicate title (Low):** "The Star-Touched Queen" appears twice at different prices
5. **Single snapshot (Low):** no time dimension, so no time trends are possible
6. **Synthetic values (Info):** uniform prices and ratings indicate a demo site

## ✅ Key conclusions

- The data is **complete but not fully clean**: category labels need fixing before use.
- Prices and ratings are **uniformly distributed**, which is a strong sign of generated demo data.
- **Price, rating, category and series status are independent.** No test found a significant link.
  Reporting "no relationship" honestly is a valid EDA outcome; forcing a story would be wrong.

## 🛠️ Tech stack
`Python` · `pandas` · `NumPy` · `SciPy` (statistics) · `Plotly` (charts) · `Streamlit` (dashboard)

## 👤 Author
**Ilma Faraz**, CodeAlpha Data Analytics Intern
