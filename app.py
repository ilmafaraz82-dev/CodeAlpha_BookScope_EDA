"""
CodeAlpha - Data Analytics - Task 2: Exploratory Data Analysis
Interactive EDA dashboard for the books dataset scraped in Task 1.

Run from cmd:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import eda

st.set_page_config(page_title="BookScope · EDA", page_icon="🔍", layout="wide")

INDIGO, SKY, GREEN, AMBER, RED, GREY = "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#CBD5E1"

# --------------------------------------------------------------------------
# Styling (same design system as Task 1)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1250px; }
.hero { background: linear-gradient(135deg, #312E81 0%, #4F46E5 55%, #0EA5E9 100%);
  border-radius: 20px; padding: 32px 38px; color: white; margin-bottom: 24px;
  box-shadow: 0 12px 30px rgba(79,70,229,.25); }
.hero .tag { display:inline-block; background: rgba(255,255,255,.18); padding: 4px 12px;
  border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: .5px; }
.hero h1 { color: white; font-size: 36px; font-weight: 800; margin: 12px 0 6px 0; }
.hero p { color: rgba(255,255,255,.9); font-size: 16px; margin: 0; max-width: 760px; }

.kpi { background: white; border: 1px solid #E5E7EB; border-radius: 16px; padding: 16px 20px; }
.kpi .label { color:#6B7280; font-size: 13px; font-weight: 500; }
.kpi .value { color:#111827; font-size: 26px; font-weight: 800; margin-top: 4px; }
.kpi .sub { color:#6B7280; font-size: 12px; margin-top: 2px; }

.card { background:white; border:1px solid #E5E7EB; border-radius:16px; padding:18px 20px; margin-bottom:14px; }
.card h4 { margin:0 0 6px 0; color:#111827; font-size:17px; }
.card p { margin:4px 0; color:#4B5563; font-size:14px; }
.qid { display:inline-block; background:#EEF2FF; color:#4F46E5; font-weight:800; border-radius:8px;
  padding:2px 10px; font-size:13px; margin-right:8px; }
.answer { background:#F0FDF4; border-left:4px solid #10B981; border-radius:8px; padding:10px 14px;
  color:#065F46; font-size:14px; margin-top:10px; }
.badge { display:inline-block; border-radius:999px; padding:3px 12px; font-size:12px; font-weight:700; }
.b-high { background:#FEE2E2; color:#B91C1C; } .b-medium { background:#FEF3C7; color:#B45309; }
.b-low { background:#E0F2FE; color:#0369A1; } .b-info { background:#F3F4F6; color:#374151; }
.b-keep { background:#F3F4F6; color:#374151; } .b-reject { background:#DCFCE7; color:#166534; }
.hyp { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin:10px 0; }
.hyp div { background:#F9FAFB; border-radius:10px; padding:10px 12px; font-size:13px; color:#374151; }
.stat { font-family: ui-monospace, monospace; background:#EEF2FF; color:#3730A3; padding:2px 8px; border-radius:6px; font-size:13px; }
.insight { background:#EEF2FF; border-radius:12px; padding:12px 16px; color:#3730A3; font-size:14px; margin:6px 0 18px 0; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background:#F3F4F6; border-radius:10px; padding:10px 16px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#4F46E5 !important; color:white !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
section[data-testid="stSidebar"] { background:#F8FAFC; }
</style>
""",
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, height=360, legend=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=46, b=10), showlegend=legend,
                      font=dict(family="Inter, sans-serif", color="#374151"),
                      plot_bgcolor="white", paper_bgcolor="white", title_font=dict(size=15, color="#111827"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


def show(fig, col=st, height=360, legend=False):
    col.plotly_chart(style(fig, height, legend), width="stretch")


@st.cache_data
def get_data(source):
    return eda.load_data(source)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Dataset")
    upload = st.file_uploader("Use a different books.csv (optional)", type="csv")
    st.caption("By default the app reads `data/books.csv`, the file produced by the Task 1 scraper.")
    st.markdown("### 🧹 Cleaning")
    hide_placeholder = st.toggle("Hide placeholder categories in charts", value=True,
                                 help="'Default' and 'Add a comment' are not real genres (see Data issues tab).")
    st.markdown("### 📏 Settings")
    st.markdown(f"Significance level **α = {eda.ALPHA}**")

if upload is None and not eda.DATA_PATH.exists():
    st.error("`data/books.csv` not found. Copy it from your Task 1 folder into a `data` folder here, or upload it in the sidebar.")
    st.stop()

df = get_data(upload if upload is not None else str(eda.DATA_PATH))
tests = eda.hypothesis_tests(df)
ans = eda.answers(df, tests)
issues = eda.data_issues(df)
ph_share = df["category"].isin(eda.PLACEHOLDER_CATEGORIES).mean()
one_book = int((df["category"].value_counts() == 1).sum())
view = df[~df["category"].isin(eda.PLACEHOLDER_CATEGORIES)] if hide_placeholder else df

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <span class="tag">CODEALPHA · DATA ANALYTICS · TASK 2</span>
  <h1>🔍 BookScope: Exploratory Data Analysis</h1>
  <p>A question-driven EDA of the <b>{len(df):,} books</b> scraped in Task 1. It explores structure, distributions,
  relationships and anomalies, tests 6 hypotheses statistically, and reports data quality problems.</p>
</div>""", unsafe_allow_html=True)

k = st.columns(5)
kpi(k[0], "Rows × columns", f"{len(df):,} × 9", "+ 4 engineered features")
kpi(k[1], "Missing values", f"{int(df.isna().sum().sum())}", "complete dataset")
kpi(k[2], "Median price", f"£{df.price_gbp.median():.2f}", f"mean £{df.price_gbp.mean():.2f}")
kpi(k[3], "Mean rating", f"{df.rating.mean():.2f} ★", f"{(df.rating >= 4).mean():.0%} rated 4-5★")
kpi(k[4], "Data issues found", f"{sum(i['Severity'] in ('High', 'Medium') for i in issues)}",
    f"+ {sum(i['Severity'] in ('Low', 'Info') for i in issues)} minor")
st.write("")

tabs = st.tabs(["❓ Questions", "🧱 Structure", "📊 Distributions", "🔗 Relationships",
                "🧪 Hypothesis tests", "⚠️ Data issues", "✅ Conclusions"])

# ==========================================================================
# 1. Questions
# ==========================================================================
with tabs[0]:
    st.markdown("#### Questions asked *before* the analysis")
    st.caption("A good EDA starts with questions, not charts. Each one is answered later with evidence.")
    show_ans = st.toggle("Reveal answers", value=False)
    c = st.columns(2)
    for i, (qid, q, why) in enumerate(eda.QUESTIONS):
        a = f'<div class="answer"><b>Answer:</b> {ans[qid]}</div>' if show_ans else ""
        c[i % 2].markdown(f'<div class="card"><h4><span class="qid">{qid}</span>{q}</h4>'
                          f'<p>🎯 {why}</p>{a}</div>', unsafe_allow_html=True)

# ==========================================================================
# 2. Structure
# ==========================================================================
with tabs[1]:
    st.markdown("#### Variables and data types")
    st.dataframe(eda.structure_table(df), hide_index=True, width="stretch", height=495)
    insight("All 9 scraped columns are complete (0 missing). <b>price_gbp</b> and <b>rating</b> are the numeric variables, "
            "<b>category</b> is the main grouping variable. Four features were engineered: title_length, is_series, "
            "price_band, category_clean.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Numeric summary")
        st.dataframe(eda.numeric_summary(df), width="stretch")
    with c2:
        st.markdown("#### Sample rows")
        st.dataframe(df[["title", "category", "price_gbp", "rating"]].sample(min(6, len(df)), random_state=1),
                     hide_index=True, width="stretch")

# ==========================================================================
# 3. Distributions (univariate)
# ==========================================================================
with tabs[2]:
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="price_gbp", nbins=25, title="Price distribution (£)", color_discrete_sequence=[INDIGO])
    fig.add_vline(df.price_gbp.mean(), line_dash="dash", line_color=AMBER,
                  annotation_text=f"mean £{df.price_gbp.mean():.2f}", annotation_position="top")
    show(fig.update_layout(bargap=0.06), c1)

    r = df["rating"].value_counts().sort_index().reset_index()
    r.columns = ["rating", "books"]
    r["share"] = (r["books"] / len(df) * 100).round(1).astype(str) + "%"
    fig = px.bar(r, x="rating", y="books", text="share", title="Books per star rating", color_discrete_sequence=[AMBER])
    fig.add_hline(len(df) / 5, line_dash="dot", line_color=GREY, annotation_text="20% if perfectly even")
    show(fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=["1★", "2★", "3★", "4★", "5★"]), c2)
    insight("Prices are <b>flat between £10 and £60</b>: no peak and no long tail. Ratings are also almost "
            "evenly spread (18-23% each). Real bookstores usually have many cheap books and mostly 4-5★ reviews, "
            "so this is our first <b>anomaly</b>: the data looks randomly generated.")

    counts = df["category"].value_counts().head(15).sort_values().reset_index()
    counts.columns = ["category", "books"]
    counts["type"] = counts["category"].isin(eda.PLACEHOLDER_CATEGORIES).map({True: "Placeholder", False: "Real genre"})
    fig = px.bar(counts, x="books", y="category", orientation="h", color="type", text="books",
                 title="Top 15 categories (placeholders in red)",
                 color_discrete_map={"Placeholder": RED, "Real genre": INDIGO})
    show(fig, height=470, legend=True)
    insight("The single biggest category is <b>'Default'</b> and the 4th is <b>'Add a comment'</b>. These are website "
            "placeholders, not genres. "
            f"Together they hold {ph_share:.0%} of all books. The categories are also very unbalanced: "
            f"{one_book} categories have only 1 book.")

    c1, c2 = st.columns(2)
    out, lo, hi = eda.iqr_outliers(df["title_length"])
    fig = px.box(df, x="title_length", points="outliers", title="Title length (characters): outliers via IQR",
                 color_discrete_sequence=[SKY], hover_data=["title"])
    show(fig, c1)
    with c2:
        st.markdown(f"**{len(out)} unusually long titles** (> {hi:.0f} characters)")
        st.dataframe(df.loc[out.index, ["title", "title_length"]].sort_values("title_length", ascending=False).head(6),
                     hide_index=True, width="stretch")
    pout, _, _ = eda.iqr_outliers(df["price_gbp"])
    st.caption(f"Price has **{len(pout)} outliers** by the IQR rule. Title-length outliers are valid long titles "
               f"(subtitles), not errors, so they are kept.")

# ==========================================================================
# 4. Relationships (bivariate / multivariate)
# ==========================================================================
with tabs[3]:
    c1, c2 = st.columns(2)
    fig = px.box(df, x="rating", y="price_gbp", title="Price by star rating", color_discrete_sequence=[INDIGO])
    show(fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], ticktext=["1★", "2★", "3★", "4★", "5★"]), c1)

    s = df.groupby("is_series")["rating"].value_counts(normalize=True).rename("share").reset_index()
    s["group"] = s["is_series"].map({True: "Series", False: "Standalone"})
    s["share"] = (s["share"] * 100).round(1)
    fig = px.bar(s, x="rating", y="share", color="group", barmode="group", title="Rating share: series vs standalone (%)",
                 color_discrete_map={"Series": SKY, "Standalone": INDIGO})
    show(fig, c2, legend=True)
    insight("The boxes for 1★ to 5★ sit at the same height, so <b>price does not rise with rating</b>. "
            "Series and standalone books also have nearly identical rating patterns.")

    big = eda.big_categories(df)
    sub = df[df["category"].isin(big)]
    order = sub.groupby("category")["price_gbp"].median().sort_values().index
    fig = px.box(sub, x="category", y="price_gbp", category_orders={"category": list(order)},
                 title=f"Price by category (categories with ≥ 30 books)", color_discrete_sequence=[GREEN])
    show(fig, height=400)

    c1, c2 = st.columns(2)
    allc = view.groupby("category").agg(books=("title", "size"), avg_price=("price_gbp", "mean")).reset_index()
    fig = px.scatter(allc, x="books", y="avg_price", hover_name="category", log_x=True, size="books",
                     title="Average price vs category size (small categories = extreme averages)",
                     color_discrete_sequence=[INDIGO])
    fig.add_hline(df.price_gbp.mean(), line_dash="dash", line_color=AMBER)
    show(fig, c1)

    ct = pd.crosstab(sub["category"], sub["rating"], normalize="index") * 100
    fig = go.Figure(go.Heatmap(z=ct.values, x=[f"{c}★" for c in ct.columns], y=ct.index,
                               colorscale=[[0, "#EEF2FF"], [1, INDIGO]], text=ct.round(0).values,
                               texttemplate="%{text}%", colorbar=dict(title="%")))
    fig.update_layout(title="Rating mix per category (% of each row)")
    show(fig, c2)
    insight("Big categories have similar price ranges. The extreme averages (e.g. a £58 or £11 category) only appear "
            "in categories with 1-3 books, a classic <b>small-sample anomaly</b>. The heatmap shows no genre "
            "is consistently rated higher.")

    corr = df[["price_gbp", "rating", "title_length"]].assign(is_series=df.is_series.astype(int)).corr(method="spearman")
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                               colorscale="RdBu", text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Spearman correlation matrix")
    c1, c2 = st.columns([1, 1])
    show(fig, c1, height=380)
    c2.markdown("<br><br>", unsafe_allow_html=True)
    c2.markdown("**How to read it:** +1 = move together, -1 = move opposite, 0 = no link. "
                "All values between the variables are close to 0, so there are **no strong relationships** in this "
                "dataset. That is an important (and honest) EDA result in itself.")

# ==========================================================================
# 5. Hypothesis tests
# ==========================================================================
with tabs[4]:
    st.markdown(f"#### Statistical tests (α = {eda.ALPHA})")
    st.caption("If p < 0.05 we reject H0 (the difference is real). Otherwise we keep H0 (the difference could be chance).")
    summary = pd.DataFrame([{"ID": t["id"], "Hypothesis": t["title"], "Test": t["test"],
                             "Statistic": t["statistic"], "p-value": round(t["p"], 3), "Decision": t["decision"]}
                            for t in tests])
    st.dataframe(summary, hide_index=True, width="stretch")
    st.write("")
    c = st.columns(2)
    for i, t in enumerate(tests):
        cls = "b-reject" if t["decision"].startswith("Reject") else "b-keep"
        c[i % 2].markdown(f"""
<div class="card">
  <h4><span class="qid">{t['id']}</span>{t['title']} <span style="color:#9CA3AF;font-size:13px">· answers {t['question']}</span></h4>
  <div class="hyp"><div><b>H0:</b> {t['H0']}</div><div><b>H1:</b> {t['H1']}</div></div>
  <p><b>Test:</b> {t['test']}. <i>{t['why']}</i></p>
  <p><span class="stat">{t['statistic']}</span> &nbsp; <span class="stat">p = {t['p']:.3f}</span> &nbsp;
     <span class="badge {cls}">{t['decision']}</span></p>
  <div class="answer">{t['meaning']}</div>
</div>""", unsafe_allow_html=True)
    insight("No test reached significance. Rather than forcing a story, the EDA concludes that <b>price, rating "
            "and category are independent</b> in this dataset, which matches a demo site with generated values.")

# ==========================================================================
# 6. Data issues
# ==========================================================================
with tabs[5]:
    st.markdown("#### Data quality report")
    st.caption("Problems to fix before using this data for modelling or business decisions.")
    for i in issues:
        cls = "b-" + i["Severity"].lower()
        st.markdown(f"""<div class="card"><h4>{i['Issue']} &nbsp;<span class="badge {cls}">{i['Severity']}</span></h4>
<p>🔎 <b>Evidence:</b> {i['Evidence']}</p><p>🛠️ <b>Fix:</b> {i['Fix']}</p></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    pc = df["category"].isin(eda.PLACEHOLDER_CATEGORIES).map({True: "Placeholder category", False: "Real genre"})
    fig = px.pie(names=pc.value_counts().index, values=pc.value_counts().values, hole=0.6,
                 title="Share of books with a usable category", color_discrete_sequence=[INDIGO, RED])
    show(fig, c1, legend=True)
    sizes = pd.cut(df["category"].value_counts(), [0, 1, 4, 29, 1000],
                   labels=["1 book", "2-4 books", "5-29 books", "30+ books"]).value_counts().sort_index()
    fig = px.bar(x=sizes.index.astype(str), y=sizes.values, text=sizes.values, title="How many categories of each size",
                 labels={"x": "", "y": "categories"}, color_discrete_sequence=[AMBER])
    show(fig, c2)

# ==========================================================================
# 7. Conclusions
# ==========================================================================
with tabs[6]:
    st.markdown("#### Answers to the questions")
    for qid, q, _ in eda.QUESTIONS:
        st.markdown(f'<div class="card"><h4><span class="qid">{qid}</span>{q}</h4>'
                    f'<div class="answer">{ans[qid]}</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class="card"><h4>🧭 Key takeaways</h4>
<p>1. The dataset is complete, but <b>{ph_share:.0%} of category labels are placeholders</b>.</p>
<p>2. Prices (£10-£60) and ratings (1-5★) are <b>evenly spread</b>, which points to generated demo data.</p>
<p>3. <b>Price, rating, category and series status are independent</b> (all 6 tests: p > 0.05).</p>
<p>4. Averages for tiny categories are misleading and should not be compared.</p></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="card"><h4>➡️ Next steps</h4>
<p>• Scrape each book's own page for real stock quantity and description (improves Task 1).</p>
<p>• Use <b>category_clean</b> and categories with ≥ 30 books in further analysis.</p>
<p>• Task 3: turn these findings into a clear visual story / dashboard.</p></div>""", unsafe_allow_html=True)
    st.download_button("⬇️ Download EDA report (Markdown)", eda.build_report(df), "eda_report.md")
