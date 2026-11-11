"""
CodeAlpha - Data Analytics - Task 2: Exploratory Data Analysis (EDA)
====================================================================
Dataset: data/books.csv  (1,000 books scraped from books.toscrape.com in Task 1)

All the analysis logic lives in this file. The Streamlit app (app.py)
only calls these functions and draws the results.

Run from cmd for a text report:
    python eda.py
This prints the report and saves it to reports/eda_report.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).parent / "data" / "books.csv"
REPORT_DIR = Path(__file__).parent / "reports"

# Categories on the website that are not real genres (found during EDA)
PLACEHOLDER_CATEGORIES = ["Default", "Add a comment"]
ALPHA = 0.05  # significance level for all hypothesis tests


# ==========================================================================
# STEP 1 - Questions we want to answer (asked BEFORE looking at results)
# ==========================================================================
QUESTIONS = [
    ("Q1", "What does the dataset contain, and is it clean enough to analyse?",
     "Bad data gives wrong conclusions, so we check quality first."),
    ("Q2", "How are book prices distributed? What is a typical price?",
     "Shows the price range a store would work with."),
    ("Q3", "How are star ratings spread? Are high ratings common?",
     "Tells us whether ratings can separate good books from bad ones."),
    ("Q4", "Which categories have the most books, and are the categories reliable?",
     "Category is the main way to group books, so it must be correct."),
    ("Q5", "Do higher-rated books cost more?",
     "If quality and price were linked, price could signal quality."),
    ("Q6", "Do some categories cost more than others?",
     "Would show whether pricing is set per genre."),
    ("Q7", "Are books in a series rated differently from standalone books?",
     "Series readers are often loyal fans, which might show in ratings."),
]


# ==========================================================================
# STEP 2 - Load the data and add useful columns (feature engineering)
# ==========================================================================
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["in_stock"] = df["in_stock"].astype(str).str.lower().eq("true")

    # New columns built from existing ones
    df["title_length"] = df["title"].str.len()                        # characters in title
    df["is_series"] = df["title"].str.contains(r"#\d", regex=True)    # e.g. "(The Hunger Games #2)"
    df["price_band"] = pd.cut(df["price_gbp"], bins=[0, 20, 30, 40, 50, 60],
                              labels=["£10-20", "£20-30", "£30-40", "£40-50", "£50-60"])
    df["category_clean"] = df["category"].where(
        ~df["category"].isin(PLACEHOLDER_CATEGORIES), "Unknown")
    return df


# ==========================================================================
# STEP 3 - Structure of the data (variables and data types)
# ==========================================================================
COLUMN_ROLES = {
    "title": "Text (identifier)", "category": "Categorical (nominal)",
    "price_gbp": "Numeric (continuous)", "rating": "Numeric (ordinal 1-5)",
    "in_stock": "Boolean", "availability": "Text", "product_url": "Text (identifier)",
    "image_url": "Text (link)", "scraped_at": "Datetime",
    "title_length": "Numeric (engineered)", "is_series": "Boolean (engineered)",
    "price_band": "Categorical (engineered)", "category_clean": "Categorical (engineered)",
}


def structure_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "Column": col,
            "Pandas type": str(df[col].dtype),
            "Variable type": COLUMN_ROLES.get(col, "-"),
            "Non-null": int(df[col].notna().sum()),
            "Missing": int(df[col].isna().sum()),
            "Unique values": int(df[col].nunique()),
            "Example": str(df[col].iloc[0])[:45],
        })
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["price_gbp", "rating", "title_length"]
    s = df[cols].describe().T
    s["skew"] = df[cols].skew()
    return s.round(2)


# ==========================================================================
# STEP 4 - Outliers / anomalies
# ==========================================================================
def iqr_outliers(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Classic IQR rule: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR."""
    q1, q3 = series.quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return series[(series < low) | (series > high)], low, high


# ==========================================================================
# STEP 5 - Data quality problems
# ==========================================================================
def data_issues(df: pd.DataFrame) -> list[dict]:
    issues = []
    placeholder = df["category"].isin(PLACEHOLDER_CATEGORIES)
    issues.append({
        "Issue": "Placeholder categories",
        "Severity": "High",
        "Evidence": f"{placeholder.sum()} books ({placeholder.mean():.0%}) are in 'Default' or "
                    f"'Add a comment'. These are not real genres.",
        "Fix": "Relabel them as 'Unknown' (done in category_clean), or scrape each book's page for a better genre.",
    })
    if df["in_stock"].nunique() == 1:
        issues.append({
            "Issue": "Column with only one value",
            "Severity": "Medium",
            "Evidence": f"in_stock is {df['in_stock'].iloc[0]} for all {len(df)} books, so it cannot explain anything.",
            "Fix": "Drop it from analysis. The listing page only shows 'In stock'. The exact quantity is on each book's own page.",
        })
    dup = df[df["title"].duplicated(keep=False)]
    if len(dup):
        issues.append({
            "Issue": "Duplicate titles",
            "Severity": "Low",
            "Evidence": f"{dup['title'].nunique()} title appears twice with different prices "
                        f"({', '.join('£%.2f' % p for p in dup['price_gbp'])}).",
            "Fix": "Kept both rows: they have different URLs, so they are probably different editions.",
        })
    counts = df["category"].value_counts()
    tiny = counts[counts < 5]
    issues.append({
        "Issue": "Very small categories",
        "Severity": "Medium",
        "Evidence": f"{len(tiny)} of {len(counts)} categories have fewer than 5 books "
                    f"({(counts == 1).sum()} have just 1). Their averages are not reliable.",
        "Fix": "Only compare categories with at least 30 books in statistical tests.",
    })
    issues.append({
        "Issue": "Single snapshot in time",
        "Severity": "Low",
        "Evidence": f"All rows were scraped at {df['scraped_at'].iloc[0]}. No time trend can be studied.",
        "Fix": "Run the Task 1 scraper on different days to build a time series.",
    })
    issues.append({
        "Issue": "Demo website / synthetic values",
        "Severity": "Info",
        "Evidence": "Prices and ratings follow almost perfect uniform distributions (see hypothesis tests H4, H5).",
        "Fix": "Treat insights as practice. Real stores would show skewed prices and more 4-5 star reviews.",
    })
    return issues


# ==========================================================================
# STEP 6 - Hypothesis testing
# ==========================================================================
def _decision(p: float) -> str:
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


def big_categories(df: pd.DataFrame, min_books: int = 30) -> list[str]:
    """Categories large enough to compare. Lowers the bar for small datasets."""
    counts = df[~df["category"].isin(PLACEHOLDER_CATEGORIES)]["category"].value_counts()
    for threshold in (min_books, 10, 5, 2):
        big = list(counts[counts >= threshold].index)
        if len(big) >= 2:
            return big
    return list(counts.index)


def _skipped(tid, question, title, reason) -> dict:
    return {"id": tid, "question": question, "title": title, "H0": "-", "H1": "-",
            "test": "Skipped", "why": reason, "statistic": "-", "p": float("nan"),
            "decision": "Not enough data", "meaning": reason}


def hypothesis_tests(df: pd.DataFrame) -> list[dict]:
    """Run all 6 tests. A test is skipped (not crashed) if the data is too small for it."""
    tests = [test_h1(df), test_h2(df), test_h3(df), test_h4(df), test_h5(df), test_h6(df)]
    return tests


def test_h1(df):
    """H1 - Do higher-rated books cost more?"""
    if df["rating"].nunique() < 2:
        return _skipped("H1", "Q5", "Higher-rated books cost more", "All books have the same rating.")
    rho, p = stats.spearmanr(df["rating"], df["price_gbp"])
    return {
        "id": "H1", "question": "Q5",
        "title": "Higher-rated books cost more",
        "H0": "There is no relationship between rating and price.",
        "H1": "Rating and price are related.",
        "test": "Spearman rank correlation",
        "why": "Rating is ordinal (1-5), so a rank-based correlation is the right choice.",
        "statistic": f"ρ = {rho:.3f}", "p": p, "decision": _decision(p),
        "meaning": (f"Correlation is {rho:.3f}, which is close to zero. A 5-star book costs about the same as a 1-star book."
                    if p >= ALPHA else f"There is a significant link (ρ = {rho:.3f})."),
    }


def test_h2(df):
    """H2 - Does price differ between categories?"""
    cats = big_categories(df)
    if len(cats) < 2:
        return _skipped("H2", "Q6", "Price differs between categories",
                        "Needs at least 2 categories. This dataset has fewer.")
    h, p = stats.kruskal(*[df.loc[df["category"] == c, "price_gbp"] for c in cats])
    return {
        "id": "H2", "question": "Q6",
        "title": "Price differs between categories",
        "H0": "All categories have the same price distribution.",
        "H1": "At least one category is priced differently.",
        "test": f"Kruskal-Wallis H-test ({len(cats)} largest categories)",
        "why": "Compares more than two groups without assuming normal data (prices are not normal).",
        "statistic": f"H = {h:.2f}", "p": p, "decision": _decision(p),
        "meaning": ("No genre is significantly more expensive. The differences we see in averages are random noise."
                    if p >= ALPHA else "Some genres are priced differently."),
    }


def test_h3(df):
    """H3 - Are series books rated differently from standalone books?"""
    s, ns = df.loc[df["is_series"], "rating"], df.loc[~df["is_series"], "rating"]
    if len(s) < 3 or len(ns) < 3:
        return _skipped("H3", "Q7", "Series books are rated differently",
                        "Needs at least 3 series and 3 standalone books.")
    u, p = stats.mannwhitneyu(s, ns)
    return {
        "id": "H3", "question": "Q7",
        "title": "Series books are rated differently",
        "H0": "Series and standalone books have the same rating distribution.",
        "H1": "Their ratings differ.",
        "test": "Mann-Whitney U test",
        "why": "Compares two independent groups on an ordinal variable.",
        "statistic": f"U = {u:,.0f}", "p": p, "decision": _decision(p),
        "meaning": (f"Series books average {s.mean():.2f}★ vs {ns.mean():.2f}★ for standalone. There is no real difference."
                    if p >= ALPHA else "Series books are rated differently."),
    }


def test_h4(df):
    """H4 - Are ratings evenly spread across 1-5 stars?"""
    observed = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    if observed.sum() < 10:
        return _skipped("H4", "Q3", "Ratings are evenly spread across 1-5 stars", "Needs at least 10 books.")
    chi, p = stats.chisquare(observed)
    return {
        "id": "H4", "question": "Q3",
        "title": "Ratings are evenly spread across 1-5 stars",
        "H0": "Each star rating is equally likely (20% each).",
        "H1": "Some ratings are more common than others.",
        "test": "Chi-square goodness-of-fit",
        "why": "Compares observed counts against expected counts.",
        "statistic": f"χ² = {chi:.2f}", "p": p, "decision": _decision(p),
        "meaning": ("Ratings look evenly spread. Real review sites usually have mostly 4-5 stars, so this suggests the ratings were randomly generated."
                    if p >= ALPHA else "Ratings are not evenly spread."),
    }


def test_h5(df):
    """H5 - Are prices uniformly distributed?"""
    lo, hi = df["price_gbp"].min(), df["price_gbp"].max()
    if len(df) < 10 or hi == lo:
        return _skipped("H5", "Q2", "Prices are uniformly distributed", "Needs at least 10 different prices.")
    d, p = stats.kstest((df["price_gbp"] - lo) / (hi - lo), "uniform")
    return {
        "id": "H5", "question": "Q2",
        "title": "Prices are uniformly distributed",
        "H0": f"Prices follow a uniform distribution between £{lo:.0f} and £{hi:.0f}.",
        "H1": "Prices follow some other distribution.",
        "test": "Kolmogorov-Smirnov test",
        "why": "Checks whether data matches a theoretical distribution.",
        "statistic": f"D = {d:.3f}", "p": p, "decision": _decision(p),
        "meaning": ("Prices are spread evenly between £10 and £60, with no peak. That is typical of generated demo data."
                    if p >= ALPHA else "Prices are not uniform."),
    }


def test_h6(df):
    """H6 - Does rating depend on category?"""
    cats = big_categories(df)
    sub = df[df["category"].isin(cats)]
    table = pd.crosstab(sub["category"], sub["rating"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _skipped("H6", "Q4", "Rating depends on category",
                        "Needs at least 2 categories and 2 rating levels.")
    chi, p, dof, _ = stats.chi2_contingency(table)
    return {
        "id": "H6", "question": "Q4",
        "title": "Rating depends on category",
        "H0": "Rating and category are independent.",
        "H1": "Some categories get better ratings.",
        "test": "Chi-square test of independence",
        "why": "Tests the link between two categorical variables.",
        "statistic": f"χ² = {chi:.2f} (dof {dof})", "p": p, "decision": _decision(p),
        "meaning": ("No category gets systematically better ratings."
                    if p >= ALPHA else "Some categories get better ratings."),
    }


# ==========================================================================
# STEP 7 - Answers to the questions
# ==========================================================================
def answers(df: pd.DataFrame, tests: list[dict]) -> dict[str, str]:
    t = {x["id"]: x for x in tests}
    counts = df["category"].value_counts()
    real = df[~df["category"].isin(PLACEHOLDER_CATEGORIES)]["category"].value_counts()
    outl, _, _ = iqr_outliers(df["price_gbp"])
    share = df["rating"].value_counts(normalize=True)
    return {
        "Q1": f"{len(df):,} rows × 9 scraped columns, no missing values. It is usable, but 22% of books have a "
              f"placeholder category and in_stock is always True (see Data issues).",
        "Q2": f"Prices run from £{df.price_gbp.min():.2f} to £{df.price_gbp.max():.2f}. The median is £{df.price_gbp.median():.2f}, "
              f"almost equal to the mean £{df.price_gbp.mean():.2f}, so there is no skew. {len(outl)} IQR outliers. "
              f"The distribution is flat (uniform, p = {t['H5']['p']:.2f}).",
        "Q3": f"Every star level holds {share.min():.0%}-{share.max():.0%} of books ({(df.rating >= 4).mean():.0%} are 4-5★). They are evenly spread "
              f"(p = {t['H4']['p']:.2f}), so ratings do not separate good and bad books well.",
        "Q4": f"The biggest real genres are {', '.join(real.index[:3])}. But the largest label overall is "
              f"'{counts.index[0]}' ({counts.iloc[0]} books), which is a placeholder, so category data needs cleaning.",
        "Q5": f"{'No' if t['H1']['p'] >= ALPHA else 'Yes'}. {t['H1']['statistic']}, p = {t['H1']['p']:.2f}.",
        "Q6": f"No significant difference among the main genres (p = {t['H2']['p']:.2f}). Extreme averages come from "
              f"tiny categories (e.g. 1-book categories).",
        "Q7": f"No. {df.is_series.sum()} series books are rated the same as standalone ones (p = {t['H3']['p']:.2f}).",
    }


# ==========================================================================
# Command-line report
# ==========================================================================
def build_report(df: pd.DataFrame) -> str:
    tests = hypothesis_tests(df)
    ans = answers(df, tests)
    lines = ["# EDA Report: Books Dataset", "", f"Rows: {len(df)} | Columns: {df.shape[1]}", ""]
    lines += ["## Questions & answers", ""]
    for qid, q, _ in QUESTIONS:
        lines += [f"**{qid}. {q}**", f"→ {ans[qid]}", ""]
    lines += ["## Hypothesis tests (α = 0.05)", "",
              "| ID | Hypothesis | Test | Statistic | p-value | Decision |", "|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['id']} | {t['title']} | {t['test']} | {t['statistic']} | {t['p']:.3f} | {t['decision']} |")
    lines += ["", "## Data issues", ""]
    for i in data_issues(df):
        lines += [f"- **{i['Issue']}** ({i['Severity']}): {i['Evidence']} *Fix:* {i['Fix']}"]
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    report = build_report(df)
    print(report)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"\nSaved: {REPORT_DIR / 'eda_report.md'}")


if __name__ == "__main__":
    main()
