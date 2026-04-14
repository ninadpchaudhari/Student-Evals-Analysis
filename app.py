import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Teaching Evaluation Analysis", layout="wide")

BLUE_DARK = "#1B4F72"
BLUE_MED = "#2E86C1"
BLUE_LIGHT = "#85C1E9"
GREEN = "#27AE60"
RED = "#E74C3C"

INSTRUCTOR_ITEMS = [
    "Instructor communicated subject clearly",
    "Instructor enthusiastic about subject",
    "Instructor created respectful atmosphere",
    "Instructor available outside class",
    "Instructor gave useful feedback",
    "Student challenged to do best work",
]

STUDENT_SELF_ITEMS = [
    "Completed assigned work before class",
    "Student came prepared for class",
    "Student contributed to class discussions",
    "Sought instructor help when needed",
]

LIKERT_COLS = [
    "Str_Disagree_pct",
    "Disagree_pct",
    "Some_Disagree_pct",
    "Neither_pct",
    "Some_Agree_pct",
    "Agree_pct",
    "Str_Agree_pct",
    "Neutral_pct",
]

LIKERT_LABELS = [
    "Strongly Disagree",
    "Disagree",
    "Somewhat Disagree",
    "Neither",
    "Somewhat Agree",
    "Agree",
    "Strongly Agree",
    "Neutral",
]

LIKERT_COLORS = [
    "#A93226",
    "#CB4335",
    "#EC7063",
    "#F4D03F",
    "#7DCEA0",
    "#52BE80",
    "#229954",
    "#95A5A6",
]

BENCHMARK_OPTIONS = {
    "Siena University": "Siena_Avg",
    "School / College": "School_Avg",
    "CSIS Department": "CSIS_Avg",
}


def extract_pct(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"(\d+)%", str(value).strip())
    return float(match.group(1)) if match else np.nan


def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def classify_item(question):
    if question in INSTRUCTOR_ITEMS:
        return "Instructor"
    if question in STUDENT_SELF_ITEMS:
        return "Student Self-Assessment"
    return "Learning Outcome"


def parse_nonstandard_excel(source):
    raw = pd.read_excel(source, header=None)
    col_names = [
        "Order",
        "Question",
        "N",
        "Avg",
        "SD",
        "Dev_from_Mean",
        "Z_Score",
        "T_Score",
        "CSIS_Avg",
        "School_Avg",
        "Siena_Avg",
        "Str_Disagree",
        "Disagree",
        "Some_Disagree",
        "Neither",
        "Some_Agree",
        "Agree",
        "Str_Agree",
        "Neutral",
    ]
    data = raw.iloc[3:].copy()
    data = data.iloc[:, : len(col_names)]
    data.columns = col_names
    data = data.dropna(subset=["Order"]).reset_index(drop=True)
    return data


def parse_csv_flexible(source):
    # Try direct CSV with usable headers first.
    direct = pd.read_csv(source)
    normalized_headers = {str(c).strip().lower(): c for c in direct.columns}
    if "order" in normalized_headers and (
        "question" in normalized_headers or "question text" in normalized_headers
    ):
        rename_map = {}
        if "question text" in normalized_headers:
            rename_map[normalized_headers["question text"]] = "Question"
        if "question" in normalized_headers:
            rename_map[normalized_headers["question"]] = "Question"
        for field in [
            "order",
            "n",
            "avg",
            "sd",
            "dev. from mean",
            "z-score",
            "t-score",
            "csis avg",
            "school avg",
            "siena avg",
            "str disagree",
            "disagree",
            "some disagree",
            "neither",
            "some agree",
            "agree",
            "str agree",
            "neutral",
        ]:
            if field in normalized_headers:
                rename_map[normalized_headers[field]] = field
        df = direct.rename(columns=rename_map)
        needed = [
            "order",
            "Question",
            "n",
            "avg",
            "sd",
            "dev. from mean",
            "z-score",
            "t-score",
            "csis avg",
            "school avg",
            "siena avg",
            "str disagree",
            "disagree",
            "some disagree",
            "neither",
            "some agree",
            "agree",
            "str agree",
            "neutral",
        ]
        if all(col in df.columns for col in needed):
            out = df[needed].copy()
            out.columns = [
                "Order",
                "Question",
                "N",
                "Avg",
                "SD",
                "Dev_from_Mean",
                "Z_Score",
                "T_Score",
                "CSIS_Avg",
                "School_Avg",
                "Siena_Avg",
                "Str_Disagree",
                "Disagree",
                "Some_Disagree",
                "Neither",
                "Some_Agree",
                "Agree",
                "Str_Agree",
                "Neutral",
            ]
            return out

    # Fall back to non-standard CSV with metadata/header rows.
    source.seek(0)
    raw = pd.read_csv(source, header=None)
    header_row = None
    for idx in range(min(8, len(raw))):
        row_vals = [str(v).strip().lower() for v in raw.iloc[idx].tolist()]
        if "order" in row_vals:
            header_row = idx
            break

    if header_row is None:
        raise ValueError("Could not detect header row in CSV file.")

    data = raw.iloc[header_row + 1 :].copy()
    data = data.iloc[:, :19]
    data.columns = [
        "Order",
        "Question",
        "N",
        "Avg",
        "SD",
        "Dev_from_Mean",
        "Z_Score",
        "T_Score",
        "CSIS_Avg",
        "School_Avg",
        "Siena_Avg",
        "Str_Disagree",
        "Disagree",
        "Some_Disagree",
        "Neither",
        "Some_Agree",
        "Agree",
        "Str_Agree",
        "Neutral",
    ]
    data = data.dropna(subset=["Order"]).reset_index(drop=True)
    return data


@st.cache_data
def load_data(uploaded_file_bytes=None, uploaded_name=None):
    if uploaded_file_bytes is None:
        default_path = Path("only_numbers.xlsx")
        if not default_path.exists():
            raise FileNotFoundError("only_numbers.xlsx not found in this folder.")
        data = parse_nonstandard_excel(default_path)
        source_used = "only_numbers.xlsx"
    else:
        ext = Path(uploaded_name).suffix.lower()
        if ext in {".xlsx", ".xls"}:
            data = parse_nonstandard_excel(pd.io.common.BytesIO(uploaded_file_bytes))
        elif ext == ".csv":
            data = parse_csv_flexible(pd.io.common.BytesIO(uploaded_file_bytes))
        else:
            raise ValueError("Unsupported file type. Please upload .xlsx or .csv")
        source_used = uploaded_name

    numeric_cols = [
        "Order",
        "N",
        "Avg",
        "SD",
        "Dev_from_Mean",
        "Z_Score",
        "T_Score",
        "CSIS_Avg",
        "School_Avg",
        "Siena_Avg",
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in [
        "Str_Disagree",
        "Disagree",
        "Some_Disagree",
        "Neither",
        "Some_Agree",
        "Agree",
        "Str_Agree",
        "Neutral",
    ]:
        data[f"{col}_pct"] = data[col].apply(extract_pct)

    data["Category"] = data["Question"].apply(classify_item)
    data["TopBox_pct"] = data["Agree_pct"].fillna(0) + data["Str_Agree_pct"].fillna(0)
    data["Unfavorable_pct"] = (
        data["Str_Disagree_pct"].fillna(0)
        + data["Disagree_pct"].fillna(0)
        + data["Some_Disagree_pct"].fillna(0)
    )

    def compute_p(row):
        t_stat = row["T_Score"]
        n = row["N"]
        if pd.isna(t_stat) or pd.isna(n) or n <= 1:
            return np.nan
        return 2 * stats.t.sf(abs(t_stat), int(n - 1))

    data["p_value"] = data.apply(compute_p, axis=1)
    data["Significance"] = data["p_value"].apply(sig_stars)

    # Use computed deviation when source field is missing.
    missing_dev = data["Dev_from_Mean"].isna()
    data.loc[missing_dev, "Dev_from_Mean"] = (
        data.loc[missing_dev, "Avg"] - data.loc[missing_dev, "Siena_Avg"]
    )

    data = data.sort_values("Order").reset_index(drop=True)
    instructor = data[data["Category"] == "Instructor"].copy()
    lo = data[data["Category"] == "Learning Outcome"].copy()

    return data, instructor, lo, source_used


def compute_t_p(your_mean, siena_mean, sd, n):
    n = int(n)
    if n <= 1 or sd <= 0:
        return np.nan, np.nan, np.nan, n - 1
    dev = your_mean - siena_mean
    se = sd / math.sqrt(n)
    if se == 0:
        return dev, se, np.nan, n - 1
    t_stat = dev / se
    p_val = 2 * stats.t.sf(abs(t_stat), n - 1)
    return dev, se, t_stat, p_val


def apply_benchmark_stats(df, benchmark_col):
    out = df.copy()
    out["Benchmark_Avg"] = out[benchmark_col]
    out["Dev_from_Benchmark"] = out["Avg"] - out["Benchmark_Avg"]

    se = out["SD"] / np.sqrt(out["N"])
    out["T_from_Benchmark"] = out["Dev_from_Benchmark"] / se
    out.loc[(out["N"] <= 1) | (out["SD"] <= 0), "T_from_Benchmark"] = np.nan

    # Keep a z-like standardized distance for display consistency.
    out["Z_from_Benchmark"] = out["Dev_from_Benchmark"] / out["SD"]
    out.loc[out["SD"] <= 0, "Z_from_Benchmark"] = np.nan

    out["p_from_Benchmark"] = out.apply(
        lambda r: 2 * stats.t.sf(abs(r["T_from_Benchmark"]), int(r["N"] - 1))
        if pd.notna(r["T_from_Benchmark"]) and pd.notna(r["N"]) and r["N"] > 1
        else np.nan,
        axis=1,
    )
    out["Sig_from_Benchmark"] = out["p_from_Benchmark"].apply(sig_stars)
    return out


def tier_learning_outcomes(lo_df, benchmark_label):
    lo_df = lo_df.copy()

    def pick_tier(row):
        dev = row["Dev_from_Benchmark"]
        p = row["p_from_Benchmark"]

        if dev > 0 and p < 0.05:
            return f"Tier 1 - Significantly Above {benchmark_label}"
        if dev < 0 and p < 0.05:
            return f"Tier 2 - Significantly Below {benchmark_label}"
        return "Tier 3 - Not Statistically Significant"

    lo_df["Tier"] = lo_df.apply(pick_tier, axis=1)
    return lo_df


def style_sig(val):
    if val == "***":
        return "background-color: #1B4F72; color: white; font-weight: bold"
    if val == "**":
        return "background-color: #2E86C1; color: white; font-weight: bold"
    if val == "*":
        return "background-color: #85C1E9; color: black; font-weight: bold"
    return "color: #5D6D7E"


def short_item_label(text, width=42):
    return text if len(text) <= width else text[: width - 3] + "..."


def widget_key(prefix, *parts):
    safe_parts = [re.sub(r"[^a-zA-Z0-9]+", "_", str(part)).strip("_") for part in parts]
    return "_".join([prefix, *safe_parts])


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def render_callout(css_class, text):
    st.markdown(
        f"""
<div class=\"{css_class}\">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def instructor_item_insight(row, benchmark_label):
    direction = "above" if row["Dev_from_Benchmark"] >= 0 else "below"
    significance = "statistically significant" if row["p_from_Benchmark"] < 0.05 else "not statistically significant"
    return (
        f"This item is {row['Dev_from_Benchmark']:+.2f} {direction} the {benchmark_label} benchmark "
        f"(t = {row['T_from_Benchmark']:.2f}, p = {fmt_p(row['p_from_Benchmark'])}) and is {significance}. "
        f"Top-box agreement is {row['TopBox_pct']:.0f}%."
    )


def instructor_explorer_insight(row, benchmark_label):
    return (
        f"With N = {int(row['N'])}, the standard error is small enough that even a deviation of "
        f"{row['Dev_from_Benchmark']:+.2f} from the {benchmark_label} mean can become statistically detectable."
    )


def learning_outcome_power_insight(row, dev, p_val):
    return (
        f"This item is {dev:+.2f} relative to benchmark at N = {int(row['N'])}. "
        f"At that sample size, the result is not significant (p = {fmt_p(p_val)}), so more observations would be needed to reduce uncertainty."
    )


def render_t_test_working(your_mean, benchmark_mean, sd, n, dev, se, t_stat, p_val, benchmark_label):
    z_value = np.nan if pd.isna(sd) or sd <= 0 else dev / sd
    st.latex(
        rf"\text{{Deviation}} = \bar{{X}} - \mu_0 = {your_mean:.2f} - {benchmark_mean:.2f} = {dev:+.2f}"
    )
    if pd.isna(z_value):
        st.latex(r"z = \text{undefined}")
    else:
        st.latex(
            rf"z = \frac{{\bar{{X}} - \mu_0}}{{s}} = \frac{{{dev:+.2f}}}{{{sd:.2f}}} = {z_value:+.2f}"
        )
    st.latex(
        rf"\text{{SE}} = \frac{{s}}{{\sqrt{{n}}}} = \frac{{{sd:.2f}}}{{\sqrt{{{int(n)}}}}} = {se:.4f}"
    )
    st.latex(
        rf"t = \frac{{\bar{{X}} - \mu_0}}{{s / \sqrt{{n}}}} = \frac{{{dev:+.2f}}}{{{se:.4f}}} = {t_stat:.2f}"
    )
    st.latex(rf"df = n - 1 = {int(n) - 1}")
    if pd.isna(p_val):
        st.latex(r"p = \text{undefined}")
    elif p_val < 0.001:
        st.latex(rf"p = 2 \cdot P(T > |t| \mid df={int(n) - 1}) < 0.001")
    else:
        st.latex(rf"p = 2 \cdot P(T > |t| \mid df={int(n) - 1}) = {p_val:.4f}")
    st.caption(f"Comparison is against the {benchmark_label} mean.")
    st.caption("The z line shows standardized distance in SD units. It is useful descriptively, but the p-value here is computed from the t-statistic because the standard error depends on sample size.")
    st.caption("Notation: lowercase t is the observed t-statistic from this item; uppercase T denotes a random variable following the t-distribution under the null hypothesis.")


def significance_color(row):
    if row["p_from_Benchmark"] < 0.001:
        return BLUE_DARK
    if row["p_from_Benchmark"] < 0.01:
        return BLUE_MED
    if row["p_from_Benchmark"] < 0.05:
        return BLUE_LIGHT
    return "#BDC3C7"


st.markdown(
    """
    <style>
    .main h1, .main h2, .main h3 { color: #1B4F72; }
    .insight {
        border-left: 5px solid #1B4F72;
        background: #F4F9FD;
        padding: 0.9rem 1rem;
        border-radius: 0.4rem;
        margin: 0.6rem 0;
    }
    .warning-note {
        border-left: 5px solid #E67E22;
        background: #FEF5E7;
        padding: 0.9rem 1rem;
        border-radius: 0.4rem;
        margin: 0.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
uploaded = st.sidebar.file_uploader(
    "Upload raw data (.xlsx or .csv)", type=["xlsx", "xls", "csv"]
)
benchmark_label = st.sidebar.selectbox(
    "Statistical benchmark",
    list(BENCHMARK_OPTIONS.keys()),
    index=0,
)
benchmark_col = BENCHMARK_OPTIONS[benchmark_label]

if uploaded is not None:
    uploaded_bytes = uploaded.getvalue()
    data, _, _, source_label = load_data(uploaded_bytes, uploaded.name)
else:
    data, _, _, source_label = load_data(None, None)

data_active = apply_benchmark_stats(data, benchmark_col)
instructor_df = data_active[data_active["Category"] == "Instructor"].copy()
lo_df = data_active[data_active["Category"] == "Learning Outcome"].copy()
lo_tiered = tier_learning_outcomes(lo_df, benchmark_label)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview & Key Findings",
        "Core Instructor Scores",
        "Learning Outcomes",
        "Interactive Significance Explorer",
        "Methodology",
    ],
)

st.sidebar.caption(f"Data source: {source_label}")
st.sidebar.caption(f"Inference baseline: {benchmark_label}")

if page == "Overview & Key Findings":
    st.title("Teaching Evaluation Analysis -- Second-Year Tenure Review")
    st.subheader("Ninad Chaudhari | Department of Computer Science | Siena University")

    eligible = 515
    core_n = int(round(instructor_df["N"].mean())) if not instructor_df.empty else 0
    lo_n = int(round(lo_tiered["N"].mean())) if not lo_tiered.empty else 0
    core_rate = (core_n / eligible * 100) if eligible else 0
    lo_rate = (lo_n / eligible * 100) if eligible else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Eligible", f"{eligible}")
    m2.metric("Core Response Rate", f"{core_rate:.1f}%", f"N ~ {core_n}")
    m3.metric("LO Response Rate", f"{lo_rate:.1f}%", f"N ~ {lo_n}")
    m4.metric("Rating Scale", "7-point Likert")

    sig_count = int((instructor_df["p_from_Benchmark"] < 0.05).sum()) if not instructor_df.empty else 0
    sig_001_count = int((instructor_df["p_from_Benchmark"] < 0.001).sum()) if not instructor_df.empty else 0
    above_count = int((instructor_df["Dev_from_Benchmark"] > 0).sum()) if not instructor_df.empty else 0
    below_count = int((instructor_df["Dev_from_Benchmark"] < 0).sum()) if not instructor_df.empty else 0
    topbox_min = instructor_df["TopBox_pct"].min() if not instructor_df.empty else np.nan
    topbox_max = instructor_df["TopBox_pct"].max() if not instructor_df.empty else np.nan

    lo_above = lo_tiered[
        lo_tiered["Tier"] == f"Tier 1 - Significantly Above {benchmark_label}"
    ].shape[0]

    st.write(
        "Across the core instructor metrics, results remain strong overall: "
        f"relative to {benchmark_label}, {above_count} items are above benchmark and {below_count} are below benchmark, with {sig_count} of {len(instructor_df)} "
        "are statistically significant at p < 0.05. "
        f"{sig_001_count} of those items reach p < 0.001, indicating highly reliable differences rather than random variation. "
        f"Top-box agreement ranges from {topbox_min:.0f}% to {topbox_max:.0f}%, and {lo_above} "
        "discipline-relevant learning outcomes also show significant positive deviations."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Core Instructor Snapshot")
        mini = instructor_df[["Question", "Avg", "Benchmark_Avg", "Sig_from_Benchmark", "TopBox_pct"]].copy()
        mini["Item"] = mini["Question"].apply(lambda x: short_item_label(x, 50))
        mini = mini[["Item", "Avg", "Benchmark_Avg", "Sig_from_Benchmark", "TopBox_pct"]]
        mini = mini.rename(
            columns={
                "Benchmark_Avg": benchmark_label,
                "Sig_from_Benchmark": "Sig",
                "TopBox_pct": "Top%",
            }
        )
        st.dataframe(
            mini.style.format({"Avg": "{:.2f}", benchmark_label: "{:.2f}", "Top%": "{:.0f}"}).map(
                style_sig, subset=["Sig"]
            ),
            use_container_width=True,
        )

    with right:
        st.markdown("### Top Learning Outcomes")
        top_lo = lo_tiered.sort_values(["Dev_from_Benchmark", "p_from_Benchmark"], ascending=[False, True]).head(3)
        top_lo = top_lo[["Question", "Avg", "Benchmark_Avg", "Dev_from_Benchmark", "p_from_Benchmark", "Sig_from_Benchmark", "TopBox_pct"]].copy()
        top_lo["Item"] = top_lo["Question"].apply(lambda x: short_item_label(x, 52))
        top_lo = top_lo[["Item", "Avg", "Benchmark_Avg", "Dev_from_Benchmark", "p_from_Benchmark", "Sig_from_Benchmark", "TopBox_pct"]]
        top_lo = top_lo.rename(
            columns={
                "Benchmark_Avg": benchmark_label,
                "Dev_from_Benchmark": "Dev",
                "p_from_Benchmark": "p-value",
                "Sig_from_Benchmark": "Sig",
                "TopBox_pct": "Top%",
            }
        )
        st.dataframe(
            top_lo.style.format({"Avg": "{:.2f}", benchmark_label: "{:.2f}", "Dev": "{:+.2f}", "p-value": fmt_p, "Top%": "{:.0f}"}).map(
                style_sig, subset=["Sig"]
            ),
            use_container_width=True,
        )

elif page == "Core Instructor Scores":
    st.title("Core Instructor Scores")

    st.markdown("### Benchmark Comparison")
    x_labels = [short_item_label(q, 32) for q in instructor_df["Question"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Chaudhari", x=x_labels, y=instructor_df["Avg"], marker_color=BLUE_DARK)
    )
    fig.add_trace(
        go.Bar(name="CSIS Dept.", x=x_labels, y=instructor_df["CSIS_Avg"], marker_color="#5DADE2")
    )
    fig.add_trace(
        go.Bar(name="School", x=x_labels, y=instructor_df["School_Avg"], marker_color="#AED6F1")
    )
    fig.add_trace(
        go.Bar(name="Siena Univ.", x=x_labels, y=instructor_df["Siena_Avg"], marker_color="#D5F5E3")
    )
    fig.update_layout(
        barmode="group",
        height=500,
        yaxis_title="Mean Score (7-point scale)",
        title="Core Instructor Evaluation Scores vs Benchmarks",
        legend_title="Group",
    )
    fig.update_yaxes(range=[5.5, 7.1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Table")
    summary = instructor_df[
        [
            "Question",
            "N",
            "Avg",
            "Benchmark_Avg",
            "Dev_from_Benchmark",
            "T_from_Benchmark",
            "p_from_Benchmark",
            "Sig_from_Benchmark",
            "TopBox_pct",
        ]
    ].copy()
    summary = summary.rename(
        columns={
            "Question": "Item",
            "Benchmark_Avg": f"{benchmark_label} Avg",
            "Dev_from_Benchmark": "Deviation",
            "T_from_Benchmark": "t-stat",
            "p_from_Benchmark": "p-value",
            "Sig_from_Benchmark": "Sig",
            "TopBox_pct": "Top-Box %",
        }
    )
    st.dataframe(
        summary.style.format(
            {
                "N": "{:.0f}",
                "Avg": "{:.2f}",
                f"{benchmark_label} Avg": "{:.2f}",
                "Deviation": "{:+.2f}",
                "t-stat": "{:.2f}",
                "p-value": fmt_p,
                "Top-Box %": "{:.0f}",
            }
        ).map(style_sig, subset=["Sig"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Item-by-Item Deep Dive")
    selected_q = st.selectbox("Select an instructor item", instructor_df["Question"].tolist())
    item = instructor_df[instructor_df["Question"] == selected_q].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg", f"{item['Avg']:.2f}")
    c2.metric(f"{benchmark_label} Avg", f"{item['Benchmark_Avg']:.2f}")
    c3.metric("Deviation", f"{item['Dev_from_Benchmark']:+.2f}")
    c4.metric("t-stat", f"{item['T_from_Benchmark']:.2f}")
    c5.metric("p-value", fmt_p(item["p_from_Benchmark"]))

    dist_vals = [item.get(col, 0) for col in LIKERT_COLS]
    dist_fig = go.Figure()
    cumulative = 0
    for label, val, color in zip(LIKERT_LABELS, dist_vals, LIKERT_COLORS):
        dist_fig.add_trace(
            go.Bar(
                y=["Response distribution"],
                x=[val],
                orientation="h",
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{x:.0f}}%<extra></extra>",
            )
        )
        cumulative += val

    dist_fig.update_layout(
        barmode="stack",
        height=240,
        xaxis_title="Percent of responses",
        showlegend=True,
        margin=dict(l=20, r=20, t=10, b=20),
    )
    st.plotly_chart(dist_fig, use_container_width=True)

    st.write(f"Top-box agreement for this item is **{item['TopBox_pct']:.0f}%**.")
    st.progress(float(np.clip(item["TopBox_pct"] / 100.0, 0, 1)))

    render_callout("insight", instructor_item_insight(item, benchmark_label))

    st.markdown(f"### Deviation from {benchmark_label} Mean")
    dev_colors = [GREEN if v >= 0 else RED for v in instructor_df["Dev_from_Benchmark"]]
    dev_fig = go.Figure(
        go.Bar(
            y=[short_item_label(x, 35) for x in instructor_df["Question"]],
            x=instructor_df["Dev_from_Benchmark"],
            orientation="h",
            marker_color=dev_colors,
            text=[f"{d:+.2f} {s}" for d, s in zip(instructor_df["Dev_from_Benchmark"], instructor_df["Sig_from_Benchmark"])],
            textposition="outside",
        )
    )
    dev_fig.update_layout(height=420, xaxis_title=f"Deviation from {benchmark_label} mean", yaxis_title="")
    st.plotly_chart(dev_fig, use_container_width=True)

    st.markdown("### t-Statistics and Significance")
    t_colors = [significance_color(r) for _, r in instructor_df.iterrows()]
    t_text = [
        f"t={t:.2f}, p={fmt_p(p)} {s}"
        for t, p, s in zip(instructor_df["T_from_Benchmark"], instructor_df["p_from_Benchmark"], instructor_df["Sig_from_Benchmark"])
    ]
    t_fig = go.Figure(
        go.Bar(
            y=[short_item_label(x, 35) for x in instructor_df["Question"]],
            x=instructor_df["T_from_Benchmark"],
            orientation="h",
            marker_color=t_colors,
            text=t_text,
            textposition="outside",
        )
    )
    t_fig.add_vline(x=1.96, line_dash="dash", line_color=RED, annotation_text="t=1.96 (p=0.05)")
    t_fig.update_layout(height=420, xaxis_title="t-statistic", yaxis_title="")
    st.plotly_chart(t_fig, use_container_width=True)

elif page == "Learning Outcomes":
    st.title("Learning Outcomes")

    st.markdown("### Three-Tier Interpretation")
    tier1 = lo_tiered[lo_tiered["Tier"] == f"Tier 1 - Significantly Above {benchmark_label}"]
    tier2 = lo_tiered[lo_tiered["Tier"] == f"Tier 2 - Significantly Below {benchmark_label}"]
    tier3 = lo_tiered[lo_tiered["Tier"] == "Tier 3 - Not Statistically Significant"]

    st.subheader(f"Tier 1 -- Significantly Above {benchmark_label}")
    st.write(
        "These outcomes align directly with disciplinary strengths in computing courses, "
        "including critical/creative thinking, quantitative skills, and scientific method."
    )
    st.dataframe(
        tier1[["Question", "N", "Avg", "Benchmark_Avg", "Dev_from_Benchmark", "p_from_Benchmark", "Sig_from_Benchmark"]]
        .rename(columns={"Benchmark_Avg": f"{benchmark_label} Avg", "Dev_from_Benchmark": "Dev", "p_from_Benchmark": "p-value", "Sig_from_Benchmark": "Sig"})
        .style.format({"Avg": "{:.2f}", f"{benchmark_label} Avg": "{:.2f}", "Dev": "{:+.2f}", "p-value": fmt_p})
        .map(style_sig, subset=["Sig"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"Tier 2 -- Significantly Below {benchmark_label}")
    st.write(
        "These outcomes are primarily institution-wide mission outcomes (for example diversity, traditions, "
        "social justice, stewardship) that are less central to most CS offerings. The pattern is typically "
        "driven by neutral responses rather than unfavorable disagreement and is often similar to CSIS averages."
    )
    st.dataframe(
        tier2[["Question", "N", "Avg", "Benchmark_Avg", "Dev_from_Benchmark", "p_from_Benchmark", "Sig_from_Benchmark"]]
        .rename(columns={"Benchmark_Avg": f"{benchmark_label} Avg", "Dev_from_Benchmark": "Dev", "p_from_Benchmark": "p-value", "Sig_from_Benchmark": "Sig"})
        .style.format({"Avg": "{:.2f}", f"{benchmark_label} Avg": "{:.2f}", "Dev": "{:+.2f}", "p-value": fmt_p})
        .map(style_sig, subset=["Sig"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Tier 3 -- Not Statistically Significant")
    st.write(
        "Several outcomes show modest differences but do not clear the p < 0.05 threshold. "
        "With LO sample sizes commonly around N=132-135, statistical power is lower than in the core instructor set."
    )
    st.dataframe(
        tier3[["Question", "N", "Avg", "Benchmark_Avg", "Dev_from_Benchmark", "p_from_Benchmark", "Sig_from_Benchmark"]]
        .rename(columns={"Benchmark_Avg": f"{benchmark_label} Avg", "Dev_from_Benchmark": "Dev", "p_from_Benchmark": "p-value", "Sig_from_Benchmark": "Sig"})
        .style.format({"Avg": "{:.2f}", f"{benchmark_label} Avg": "{:.2f}", "Dev": "{:+.2f}", "p-value": fmt_p})
        .map(style_sig, subset=["Sig"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Filterable LO Table")
    tier_choice = st.selectbox(
        "Filter by tier",
        [
            "All",
            f"Tier 1 - Significantly Above {benchmark_label}",
            f"Tier 2 - Significantly Below {benchmark_label}",
            "Tier 3 - Not Statistically Significant",
        ],
    )
    filtered = lo_tiered if tier_choice == "All" else lo_tiered[lo_tiered["Tier"] == tier_choice]

    show = filtered[
        [
            "Question",
            "Tier",
            "N",
            "Avg",
            "Benchmark_Avg",
            "Dev_from_Benchmark",
            "T_from_Benchmark",
            "p_from_Benchmark",
            "Sig_from_Benchmark",
            "TopBox_pct",
        ]
    ].rename(
        columns={
            "Benchmark_Avg": f"{benchmark_label} Avg",
            "Dev_from_Benchmark": "Dev",
            "T_from_Benchmark": "t-stat",
            "p_from_Benchmark": "p-value",
            "Sig_from_Benchmark": "Sig",
            "TopBox_pct": "Top-Box%",
        }
    )
    st.dataframe(
        show.style.format(
            {
                "N": "{:.0f}",
                "Avg": "{:.2f}",
                f"{benchmark_label} Avg": "{:.2f}",
                "Dev": "{:+.2f}",
                "t-stat": "{:.2f}",
                "p-value": fmt_p,
                "Top-Box%": "{:.0f}",
            }
        ).map(style_sig, subset=["Sig"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Learning Outcomes Deviation Chart")
    lo_sorted = lo_tiered.sort_values("Dev_from_Benchmark")
    lo_dev_fig = px.bar(
        lo_sorted,
        x="Dev_from_Benchmark",
        y="Question",
        orientation="h",
        color="Dev_from_Benchmark",
        color_continuous_scale=[(0, RED),  (1, GREEN)],
        hover_data={"N": True, "Avg": ":.2f", "Benchmark_Avg": ":.2f", "p_from_Benchmark": ":.4f"},
    )
    lo_dev_fig.update_layout(height=700, coloraxis_showscale=False, xaxis_title=f"Deviation from {benchmark_label}")
    st.plotly_chart(lo_dev_fig, use_container_width=True)

    st.markdown("### Learning Outcomes Top-Box Chart")
    lo_top_fig = px.bar(
        lo_tiered.sort_values("TopBox_pct"),
        x="TopBox_pct",
        y="Question",
        orientation="h",
        color="TopBox_pct",

        hover_data={"N": True, "Avg": ":.2f", "Benchmark_Avg": ":.2f"},
    )
    lo_top_fig.update_layout(height=700, coloraxis_showscale=False, xaxis_title="Top-Box (%)")
    st.plotly_chart(lo_top_fig, use_container_width=True)

elif page == "Interactive Significance Explorer":
    st.title("Interactive Significance Explorer")
    st.write(
        "Adjust means, standard deviations, and sample size to see exactly how statistical significance changes. "
        "This makes the one-sample t-test mechanics visible for both core instructor metrics and learning outcomes."
    )

    tab_a, tab_b = st.tabs(["Instructor Quality Metrics", "Learning Outcome Metrics"])

    with tab_a:
        sel = st.selectbox("Instructor item", instructor_df["Question"].tolist(), key="ins_sel")
        row = instructor_df[instructor_df["Question"] == sel].iloc[0]
        ins_scope = (sel, benchmark_label)

        c1, c2, c3, c4 = st.columns(4)
        your_mean = c1.number_input("Your Mean", min_value=1.0, max_value=7.0, value=float(row["Avg"]), step=0.01, key=widget_key("ins_mean", *ins_scope))
        siena_mean = c2.number_input(f"{benchmark_label} Mean", min_value=1.0, max_value=7.0, value=float(row["Benchmark_Avg"]), step=0.01, key=widget_key("ins_benchmark", *ins_scope))
        sd = c3.number_input("Standard Deviation", min_value=0.01, max_value=3.0, value=float(row["SD"]), step=0.01, key=widget_key("ins_sd", *ins_scope))
        n = c4.number_input("Sample Size N", min_value=2, max_value=2000, value=int(row["N"]), step=1, key=widget_key("ins_n", *ins_scope))

        dev, se, t_stat, p_val = compute_t_p(your_mean, siena_mean, sd, n)

        st.latex(r"t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}")
        st.latex(r"z = \frac{\bar{X} - \mu_0}{s}")
        st.latex(r"\text{SE} = \frac{s}{\sqrt{n}}")
        st.latex(r"p = 2 \cdot P(T > |t| \;;\; df = n - 1)")
        st.caption("Here, z is a standardized distance from benchmark in SD units, while t is the inferential statistic that divides by the standard error. T is the theoretical t-distribution used to measure how extreme that observed t is under the null hypothesis.")

        render_t_test_working(your_mean, siena_mean, sd, n, dev, se, t_stat, p_val, benchmark_label)

        is_sig = bool(p_val < 0.05)
        st.markdown(
            f"### Significant at p < 0.05? {'YES' if is_sig else 'NO'}",
            help="Two-tailed one-sample t-test decision rule.",
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(t_stat),
                title={"text": "Computed t-statistic"},
                gauge={
                    "axis": {"range": [-12, 12]},
                    "bar": {"color": BLUE_DARK},
                    "steps": [
                        {"range": [-1.96, 1.96], "color": "#FADBD8"},
                        {"range": [-12, -1.96], "color": "#D6EAF8"},
                        {"range": [1.96, 12], "color": "#D6EAF8"},
                    ],
                    "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.75, "value": 1.96},
                },
            )
        )
        gauge.update_layout(height=280)
        st.plotly_chart(gauge, use_container_width=True)

        render_callout("insight", instructor_explorer_insight(row, benchmark_label))

    with tab_b:
        lo_pick = st.selectbox("Learning outcome item", lo_tiered["Question"].tolist(), key="lo_sel")
        lo_row = lo_tiered[lo_tiered["Question"] == lo_pick].iloc[0]
        lo_scope = (lo_pick, benchmark_label)

        c1, c2, c3, c4 = st.columns(4)
        your_mean_lo = c1.number_input("Your Mean", min_value=1.0, max_value=7.0, value=float(lo_row["Avg"]), step=0.01, key=widget_key("lo_mean", *lo_scope))
        siena_mean_lo = c2.number_input(f"{benchmark_label} Mean", min_value=1.0, max_value=7.0, value=float(lo_row["Benchmark_Avg"]), step=0.01, key=widget_key("lo_benchmark", *lo_scope))
        sd_lo = c3.number_input("Standard Deviation", min_value=0.01, max_value=3.0, value=float(lo_row["SD"]), step=0.01, key=widget_key("lo_sd", *lo_scope))
        n_lo = c4.number_input("Sample Size N", min_value=2, max_value=2000, value=int(lo_row["N"]), step=1, key=widget_key("lo_n", *lo_scope))

        dev_lo, se_lo, t_lo, p_lo = compute_t_p(your_mean_lo, siena_mean_lo, sd_lo, n_lo)

        render_t_test_working(your_mean_lo, siena_mean_lo, sd_lo, n_lo, dev_lo, se_lo, t_lo, p_lo, benchmark_label)

        chosen_n = st.slider("What if N were larger?", min_value=50, max_value=500, value=max(50, min(500, int(lo_row["N"]))), step=5)

        n_vals = np.arange(50, 501)
        dev_fixed = your_mean_lo - siena_mean_lo
        p_series = [
            2 * stats.t.sf(abs(dev_fixed / (sd_lo / math.sqrt(int(nv)))), int(nv - 1))
            for nv in n_vals
        ]
        power_df = pd.DataFrame({"N": n_vals, "p_value": p_series})

        if sd_lo > 0 and chosen_n > 1:
            _, _, t_hyp, p_hyp = compute_t_p(your_mean_lo, siena_mean_lo, sd_lo, chosen_n)
            st.write(f"At N = {chosen_n}, the same deviation would yield t = {t_hyp:.2f} and p = {fmt_p(p_hyp)}.")

        power_fig = px.line(power_df, x="N", y="p_value", title="How p-value changes as N increases")
        power_fig.add_hline(y=0.05, line_color=RED, line_dash="dash", annotation_text="p=0.05")
        actual_n = int(lo_row["N"])
        actual_p = 2 * stats.t.sf(abs(lo_row["T_from_Benchmark"]), int(actual_n - 1))
        power_fig.add_trace(
            go.Scatter(
                x=[actual_n],
                y=[actual_p],
                mode="markers",
                marker=dict(size=11, color=BLUE_DARK),
                name="Actual N",
                hovertemplate=f"Actual N={actual_n}<br>p={actual_p:.4f}<extra></extra>",
            )
        )
        power_fig.update_layout(height=380, yaxis_title="p-value")
        st.plotly_chart(power_fig, use_container_width=True)

        if p_lo >= 0.05 and dev_lo > 0:
            render_callout("insight", learning_outcome_power_insight(lo_row, dev_lo, p_lo))
        if str(lo_row["Tier"]).startswith("Tier 2 - Significantly Below"):
            render_callout(
                "warning-note",
                "This outcome sits in the significantly-below tier. For many CS courses, that often reflects curriculum emphasis rather than instructional weakness, especially for broad institution-level outcomes.",
            )

    st.markdown("### Side-by-Side Item Comparison")
    compare_df = pd.concat([instructor_df, lo_tiered], ignore_index=True)
    all_items = compare_df["Question"].tolist()
    left_sel, right_sel = st.columns(2)
    item_a = left_sel.selectbox("Item A", all_items, index=0, key="cmp_a")
    item_b = right_sel.selectbox("Item B", all_items, index=min(1, len(all_items) - 1), key="cmp_b")

    a_row = compare_df[compare_df["Question"] == item_a].iloc[0]
    b_row = compare_df[compare_df["Question"] == item_b].iloc[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Item A:** {item_a}")
        st.write(
            f"Avg={a_row['Avg']:.2f}, {benchmark_label}={a_row['Benchmark_Avg']:.2f}, "
            f"Dev={a_row['Dev_from_Benchmark']:+.2f}, t={a_row['T_from_Benchmark']:.2f}, p={fmt_p(a_row['p_from_Benchmark'])}, Sig={a_row['Sig_from_Benchmark']}"
        )
    with col_b:
        st.markdown(f"**Item B:** {item_b}")
        st.write(
            f"Avg={b_row['Avg']:.2f}, {benchmark_label}={b_row['Benchmark_Avg']:.2f}, "
            f"Dev={b_row['Dev_from_Benchmark']:+.2f}, t={b_row['T_from_Benchmark']:.2f}, p={fmt_p(b_row['p_from_Benchmark'])}, Sig={b_row['Sig_from_Benchmark']}"
        )

    with st.expander("Formula reference"):
        st.markdown(
            r"""
One-Sample t-Test

H0: mu_instructor = mu_benchmark

H1: mu_instructor != mu_benchmark (two-tailed)

$$
 z = \frac{\bar{X} - \mu_0}{s}
$$

$$
 t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}}
$$

$$
 df = n - 1
$$

$$
 p = 2 \cdot P(T > |t|; df)
$$

Reject H0 if p < 0.05.
            """
        )
    st.write("The z-score is a descriptive standardized distance from the benchmark, measured in SD units. The t-statistic is the inferential quantity because it additionally accounts for sample size through the standard error. Uppercase T refers to the t-distribution as a random variable under the null hypothesis, while lowercase t is the observed test statistic from the sample.")

elif page == "Methodology":
    st.title("Methodology")

    st.markdown("### Research Questions")
    st.write(f"RQ1: Do core instructor evaluation scores exceed {benchmark_label} benchmarks?")
    st.write(f"RQ2: Which learning outcomes are above, below, or statistically indistinguishable from {benchmark_label} means?")
    st.write("RQ3: How should statistical results be interpreted in light of curriculum alignment and response patterns?")

    st.markdown("### Hypotheses")
    st.write(f"For each item, the one-sample test evaluates whether the observed course mean differs from the {benchmark_label} benchmark.")
    st.latex(r"H_0: \mu_{course} = \mu_{benchmark}")
    st.latex(r"H_1: \mu_{course} \neq \mu_{benchmark}")

    st.markdown("### One-Sample t-Test")
    st.latex(r"t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}}")
    st.write(
        "The t-statistic scales the observed deviation by its standard error. Larger absolute t-values indicate stronger evidence against the null hypothesis."
    )
    st.write("Notation matters here: lowercase t is the observed value computed from the course data, while uppercase T in expressions like p = 2 · P(T > |t|) refers to the reference t-distribution used to compute tail probability under the null model.")

    st.markdown("### Z-Score vs T-Statistic")
    st.write(
        "The SmartEvals output includes both Z-scores and T-scores. The T-statistic directly incorporates sample size through the standard error term and is the appropriate quantity for significance testing in this context."
    )
    st.latex(r"z = \frac{\bar{X} - \mu_0}{s}")
    st.latex(r"t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}} = z \cdot \sqrt{n}")
    st.write("Use z to describe how far the course mean is from the benchmark in SD units. Use t for significance testing, because t scales that same deviation by the standard error and therefore reflects sample size. In other words, the reported t-statistic is a realized sample value, and the t-distribution, written with a capital T in the p-value formula, is the probability model we compare that realized value against.")

    st.markdown("### What Significance Means (and Does Not Mean)")
    st.write(
        "Statistical significance indicates that an observed difference is unlikely to be random noise under the null model. "
        "It does not, by itself, measure pedagogical importance or effect size. Practical interpretation should combine magnitude, "
        "response distribution, and curricular context."
    )

    st.markdown("### Why Top-Box Complements t-Tests")
    st.write(
        "Top-box percentages (Agree + Strongly Agree) provide an intuitive measure of endorsement intensity. "
        "An item can be statistically significant with a small mean difference if N is large; top-box values help contextualize real-world impact."
    )

    st.markdown("### Sample Size and Power")
    instructor_mean_n = int(round(instructor_df["N"].mean())) if not instructor_df.empty else 0
    lo_mean_n = int(round(lo_df["N"].mean())) if not lo_df.empty else 0
    st.write(
        f"Core instructor items have much larger sample sizes (around N={instructor_mean_n}) than most learning outcomes "
        f"(around N={lo_mean_n}). This difference increases statistical power for instructor items and explains why similarly sized deviations can have different p-values."
    )

    st.markdown("### Assumptions and Limitations")
    st.write("Responses are treated as approximately interval-scaled for mean-based analysis.")
    st.write("Each item is tested independently; no multiple-comparison adjustment is applied in this dashboard.")
    st.write("Interpretation of mission-level learning outcomes should account for disciplinary fit and course focus.")
