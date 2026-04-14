import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import re

st.set_page_config(page_title="Teaching Evaluation Analysis", layout="wide", page_icon="📊")

# --- DATA LOADING & PARSING ---
@st.cache_data
def load_data(file_obj, is_csv=False):
    if is_csv:
        raw = pd.read_csv(file_obj, header=None)
        col_names = ['Order', 'Question', 'N', 'Avg', 'SD', 'CSIS_Avg', 'School_Avg', 'Siena_Avg',
                     'Str_Disagree', 'Disagree', 'Some_Disagree', 'Neither', 'Some_Agree', 'Agree', 'Str_Agree', 'Neutral']
    else:
        raw = pd.read_excel(file_obj, header=None)
        col_names = ['Order', 'Question', 'N', 'Avg', 'SD', 'Dev_from_Mean', 
                     'Z_Score', 'T_Score', 'CSIS_Avg', 'School_Avg', 'Siena_Avg',
                     'Str_Disagree', 'Disagree', 'Some_Disagree', 'Neither',
                     'Some_Agree', 'Agree', 'Str_Agree', 'Neutral']

    data = raw.iloc[3:].copy()
    data.columns = col_names
    data = data.dropna(subset=['Order']).reset_index(drop=True)

    numeric_cols = ['Order', 'N', 'Avg', 'SD', 'CSIS_Avg', 'School_Avg', 'Siena_Avg']
    if not is_csv:
        numeric_cols.extend(['Dev_from_Mean', 'Z_Score', 'T_Score'])

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    if is_csv:
        data['Dev_from_Mean'] = data['Avg'] - data['Siena_Avg']
        se = data['SD'] / np.sqrt(data['N'])
        data['T_Score'] = data['Dev_from_Mean'] / se

    def extract_pct(val):
        if pd.isna(val): return np.nan
        m = re.match(r'(\d+)%', str(val).strip())
        return int(m.group(1)) if m else np.nan

    for col in ['Str_Disagree', 'Disagree', 'Some_Disagree', 'Neither', 
                'Some_Agree', 'Agree', 'Str_Agree', 'Neutral']:
        data[col + '_pct'] = data[col].apply(extract_pct)

    instructor_questions = [
        'Instructor communicated subject clearly',
        'Instructor enthusiastic about subject',
        'Instructor created respectful atmosphere',
        'Instructor available outside class',
        'Instructor gave useful feedback',
        'Student challenged to do best work'
    ]
    student_self_questions = [
        'Completed assigned work before class',
        'Student came prepared for class',
        'Student contributed to class discussions',
        'Sought instructor help when needed'
    ]

    def classify(q):
        if q in instructor_questions: return 'Instructor'
        elif q in student_self_questions: return 'Student Self-Assessment'
        else: return 'Learning Outcome'

    data['Category'] = data['Question'].apply(classify)
    data['TopBox_pct'] = data['Agree_pct'] + data['Str_Agree_pct']
    data['Unfavorable_pct'] = data['Str_Disagree_pct'] + data['Disagree_pct'] + data['Some_Disagree_pct']

    def compute_p_value(row):
        t_stat = row['T_Score']
        n = row['N']
        if pd.isna(t_stat) or pd.isna(n) or n <= 1:
            return np.nan
        df = n - 1
        return 2 * stats.t.sf(abs(t_stat), df)

    data['p_value'] = data.apply(compute_p_value, axis=1)

    def sig_stars(p):
        if pd.isna(p): return ''
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return 'n.s.'

    data['Significance'] = data['p_value'].apply(sig_stars)
    return data

# Setup sidebar for file upload
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Evaluation Data", type=['xlsx', 'csv'])

# Default file logic
if uploaded_file is not None:
    is_csv = uploaded_file.name.endswith('.csv')
    df = load_data(uploaded_file, is_csv=is_csv)
else:
    # Use default if available
    try:
        df = load_data('only_numbers.xlsx', is_csv=False)
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a file to proceed.")
        st.stop()

# Build Data Subsets
df_inst = df[df['Category'] == 'Instructor'].copy()
df_lo = df[df['Category'] == 'Learning Outcome'].copy()

# Add LO Tiers
def get_lo_tier(row):
    if row['Significance'] in ['*', '**', '***'] and row['Dev_from_Mean'] > 0:
        return 'Tier 1 - Significantly Above Siena'
    elif row['Significance'] in ['*', '**', '***'] and row['Dev_from_Mean'] < 0:
        return 'Tier 2 - Significantly Below Siena'
    else:
        return 'Tier 3 - Not Significant'

df_lo['Tier'] = df_lo.apply(get_lo_tier, axis=1)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview & Key Findings", 
    "Core Instructor Scores", 
    "Learning Outcomes", 
    "Interactive Significance Explorer", 
    "Methodology"
])

# --- HELPER FUNCS ---
def format_p(p):
    if pd.isna(p): return "N/A"
    return f"{p:.4f}" if p >= 0.001 else "<0.001"

# --- PAGE 1: OVERVIEW & KEY FINDINGS ---
if page == "Overview & Key Findings":
    st.title("Teaching Evaluation Analysis")
    st.subheader("Ninad Chaudhari | Department of Computer Science | Siena University")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Eligible", "515")
    with col2:
        st.metric("Core Response Rate", "~76%")
    with col3:
        st.metric("LO Response Rate", "~26%")
    with col4:
        st.metric("Scale", "7-Point Likert")

    st.markdown("### Key Findings")
    st.markdown('''
    - **Strong Pedagogical Delivery:** All 6 core instructor items exceeded the Siena University average.
    - **High Statistical Confidence:** All 6 instructor items are statistically significant, with 5 of 6 reaching $p < 0.001$.
    - **Excellent Practical Significance:** Top-box favorability (Agree + Strongly Agree) ranged from 82% to 99% for instructor items.
    - **Curriculum Alignment:** 3 discipline-relevant learning outcomes (Critical/creative thinking, Quantitative skills, Scientific method) were significantly above the institutional average, validating strong alignment with departmental goals.
    ''')

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Core Instructor Summary")
        disp_inst = df_inst[['Question', 'Avg', 'Siena_Avg', 'Significance', 'TopBox_pct']]
        st.dataframe(disp_inst.style.format({"Avg": "{:.2f}", "Siena_Avg": "{:.2f}", "TopBox_pct": "{:.0f}%"}), use_container_width=True)

    with colB:
        st.markdown("#### Top 3 Learning Outcomes")
        top_los = df_lo[df_lo['Tier'] == 'Tier 1 - Significantly Above Siena'].sort_values('Dev_from_Mean', ascending=False)
        disp_lo = top_los[['Question', 'Avg', 'Siena_Avg', 'Significance', 'TopBox_pct']]
        st.dataframe(disp_lo.style.format({"Avg": "{:.2f}", "Siena_Avg": "{:.2f}", "TopBox_pct": "{:.0f}%"}), use_container_width=True)


# --- PAGE 2: CORE INSTRUCTOR SCORES ---
elif page == "Core Instructor Scores":
    st.title("Core Instructor Scores")
    
    st.markdown("### Benchmark Comparison")
    labels = [q.replace('Instructor ', '').replace('Student ', '') for q in df_inst['Question']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=df_inst['Avg'], name='Chaudhari', marker_color='#1B4F72'))
    fig.add_trace(go.Bar(x=labels, y=df_inst['CSIS_Avg'], name='CSIS Dept.', marker_color='#5DADE2'))
    fig.add_trace(go.Bar(x=labels, y=df_inst['School_Avg'], name='School', marker_color='#AED6F1'))
    fig.add_trace(go.Bar(x=labels, y=df_inst['Siena_Avg'], name='Siena Univ.', marker_color='#D5F5E3'))
    
    fig.update_layout(
        barmode='group',
        yaxis=dict(title='Mean Score (7-point scale)', range=[5.5, 7.1]),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Data")
    tab_inst = df_inst[['Question', 'N', 'Avg', 'Siena_Avg', 'Dev_from_Mean', 'T_Score', 'p_value', 'Significance', 'TopBox_pct']].copy()
    st.dataframe(tab_inst.style.format({
        "Avg": "{:.2f}", "Siena_Avg": "{:.2f}", "Dev_from_Mean": "{:+.2f}",
        "T_Score": "{:.2f}", "p_value": "{:.4f}", "TopBox_pct": "{:.0f}%"
    }), use_container_width=True)

    st.markdown("---")
    st.markdown("### Item-by-Item Deep Dive")
    selected_item = st.selectbox("Select an evaluation item:", df_inst['Question'].tolist())
    item_data = df_inst[df_inst['Question'] == selected_item].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Instruct. Avg", f"{item_data['Avg']:.2f}")
    col2.metric("Siena Avg", f"{item_data['Siena_Avg']:.2f}")
    col3.metric("Deviation", f"{item_data['Dev_from_Mean']:+.2f}")
    col4.metric("t-stat", f"{item_data['T_Score']:.2f}")
    col5.metric("p-value", f"{item_data['p_value']:.4f} {item_data['Significance']}")

    # Stacked bar
    likert_cols = ['Str_Disagree_pct', 'Disagree_pct', 'Some_Disagree_pct', 'Neither_pct', 'Some_Agree_pct', 'Agree_pct', 'Str_Agree_pct']
    likert_names = ['Str Disagree', 'Disagree', 'Some Disagree', 'Neither', 'Some Agree', 'Agree', 'Str Agree']
    colors = ['#E74C3C', '#E67E22', '#F1C40F', '#BDC3C7', '#A9DFBF', '#27AE60', '#117A65']
    
    fig_stack = go.Figure()
    for col, name, color in zip(likert_cols, likert_names, colors):
        val = item_data[col]
        if not pd.isna(val) and val > 0:
            fig_stack.add_trace(go.Bar(
                y=['Responses'], x=[val], name=name, orientation='h',
                marker=dict(color=color),
                text=f"{val}%", textposition='inside'
            ))
    fig_stack.update_layout(barmode='stack', height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
    st.plotly_chart(fig_stack, use_container_width=True)
    
    st.progress(item_data['TopBox_pct']/100.0, text=f"Top-Box Favorability: {item_data['TopBox_pct']}%")

    st.markdown("---")
    colC, colD = st.columns(2)
    with colC:
        st.markdown("#### Deviation from Siena Mean")
        fig_dev = go.Figure()
        colors_dev = ['#27AE60' if x > 0 else '#E74C3C' for x in df_inst['Dev_from_Mean']]
        fig_dev.add_trace(go.Bar(
            y=df_inst['Question'], x=df_inst['Dev_from_Mean'], 
            orientation='h', marker_color=colors_dev,
            text=df_inst['Significance'], textposition='outside'
        ))
        fig_dev.update_layout(height=400, margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_dev, use_container_width=True)

    with colD:
        st.markdown("#### T-Statistic vs Significance Threshold")
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            y=df_inst['Question'], x=df_inst['T_Score'], 
            orientation='h', marker_color='#1B4F72',
            text=[f"t={t:.2f} {s}" for t, s in zip(df_inst['T_Score'], df_inst['Significance'])], textposition='outside'
        ))
        fig_t.add_vline(x=1.96, line_dash="dash", line_color="red", annotation_text="p=0.05 threshold")
        fig_t.update_layout(height=400, margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_t, use_container_width=True)


# --- PAGE 3: LEARNING OUTCOMES ---
elif page == "Learning Outcomes":
    st.title("Learning Outcomes")

    # Tiered Explanations
    st.markdown("### Tier 1 — Significantly Above Siena")
    st.info("These 3 items reflect strong pedagogical delivery in areas core to the discipline (Critical thinking, Quantitative skills, Scientific method).")
    
    st.markdown("### Tier 2 — Significantly Below Siena")
    st.warning("These 10 items relate to institutional core goals (diversity, social justice, historical traditions). The negative deviation reflects the curriculum focus of a technical CS course, rather than teaching quality. High 'Neutral' responses drive these figures.")
    
    st.markdown("### Tier 3 — Not Significant")
    st.error("These 8 items showed differences that were not statistically significant given the moderate sample size ($N \approx 135$).")

    st.markdown("---")
    
    tier_filter = st.selectbox("Filter by Tier", ['All'] + list(df_lo['Tier'].unique()))
    if tier_filter != 'All':
        df_lo_filtered = df_lo[df_lo['Tier'] == tier_filter]
    else:
        df_lo_filtered = df_lo
        
    st.dataframe(df_lo_filtered[['Question', 'N', 'Avg', 'Siena_Avg', 'Dev_from_Mean', 'T_Score', 'p_value', 'Significance', 'TopBox_pct']].style.format({
        "Avg": "{:.2f}", "Siena_Avg": "{:.2f}", "Dev_from_Mean": "{:+.2f}",
        "T_Score": "{:.2f}", "p_value": "{:.4f}", "TopBox_pct": "{:.0f}%"
    }), use_container_width=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Deviation from Mean (All LOs)")
        sort_lo = df_lo.sort_values('Dev_from_Mean', ascending=True)
        colors_lo = ['#27AE60' if x > 0 else '#E74C3C' for x in sort_lo['Dev_from_Mean']]
        
        # Abbreviate long labels for chart
        short_labels = [l[:40]+"..." if len(l)>40 else l for l in sort_lo['Question']]
        
        fig_lo_dev = go.Figure()
        fig_lo_dev.add_trace(go.Bar(
            y=short_labels, x=sort_lo['Dev_from_Mean'], 
            orientation='h', marker_color=colors_lo,
            text=sort_lo['Significance'], textposition='outside',
            hovertext=sort_lo['Question']
        ))
        fig_lo_dev.update_layout(height=600, margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_lo_dev, use_container_width=True)

    with colB:
        st.markdown("#### Top-Box % (All LOs)")
        sort_top = df_lo.sort_values('TopBox_pct', ascending=True)
        short_labels_top = [l[:40]+"..." if len(l)>40 else l for l in sort_top['Question']]
        
        fig_lo_top = go.Figure()
        fig_lo_top.add_trace(go.Bar(
            y=short_labels_top, x=sort_top['TopBox_pct'], 
            orientation='h', marker_color='#1B4F72',
            text=[f"{v:.0f}%" for v in sort_top['TopBox_pct']], textposition='outside',
            hovertext=sort_top['Question']
        ))
        fig_lo_top.update_layout(height=600, margin=dict(l=10, r=20, t=10, b=20))
        st.plotly_chart(fig_lo_top, use_container_width=True)


# --- PAGE 4: INTERACTIVE SIGNIFICANCE EXPLORER ---
elif page == "Interactive Significance Explorer":
    st.title("Interactive Significance Explorer")
    st.markdown("This tool demonstrates the relationship between Sample Size ($N$), Mean Deviation, and Statistical Significance (p-value).")
    
    with st.expander("Show Statistical Formulas"):
        st.latex(r"t = \frac{\bar{X} - \mu_0}{s / \sqrt{n}} \quad \text{where} \quad \text{SE} = \frac{s}{\sqrt{n}}")
        st.latex(r"p = 2 \cdot P(T > |t| \;;\; df = n - 1)")
        st.markdown("Reject $H_0$ (declare significance) if $p < 0.05$")

    tab1, tab2 = st.tabs(["Instructor Quality Metrics", "Learning Outcome Metrics"])

    # --- TAB 1: INSTRUCTOR ---
    with tab1:
        st.markdown("### Explore Core Instructor Items")
        item_sel_inst = st.selectbox("Select an item to pre-fill actual values:", df_inst['Question'].tolist(), key="inst_sel")
        actual = df_inst[df_inst['Question'] == item_sel_inst].iloc[0]

        st.info("With $N \approx 392$, even a small deviation of $+0.12$ is significant because the standard error shrinks to $SD/\sqrt{N}$. Fast sample sizes give us the power to detect small real differences.")

        col1, col2, col3, col4 = st.columns(4)
        c_mean = col1.number_input("Your Mean", value=float(actual['Avg']), step=0.05, key="inst_mean")
        siena_mean = col2.number_input("Siena Mean", value=float(actual['Siena_Avg']), step=0.05, key="inst_siena")
        sd = col3.number_input("Standard Deviation", value=float(actual['SD']), step=0.05, key="inst_sd")
        n_val = col4.number_input("Sample Size N", value=int(actual['N']), step=10, min_value=2, key="inst_n")

        # Computations
        dev = c_mean - siena_mean
        se = sd / np.sqrt(n_val)
        t_stat = dev / se
        df_stat = n_val - 1
        p_val = 2 * stats.t.sf(abs(t_stat), df_stat)

        st.markdown("---")
        st.markdown(f"**Deviation** = {c_mean:.2f} - {siena_mean:.2f} = `{dev:+.2f}`")
        st.markdown(f"**Standard Error** = {sd:.2f} / sqrt({n_val}) = `{se:.4f}`")
        st.markdown(f"**t-statistic** = {dev:+.2f} / {se:.4f} = `{t_stat:.2f}`")
        st.markdown(f"**degrees of freedom** = {n_val} - 1 = `{df_stat}`")
        st.markdown(f"**p-value** = `{p_val:.4f}`")

        is_sig = p_val < 0.05
        if is_sig:
            st.success(f"Significant at p < 0.05? **YES**")
        else:
            st.error(f"Significant at p < 0.05? **NO**")
            
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = abs(t_stat),
            title = {'text': "|t-statistic|"},
            gauge = {
                'axis': {'range': [0, 10]},
                'bar': {'color': "#1B4F72"},
                'steps': [
                    {'range': [0, 1.96], 'color': "#E74C3C"},
                    {'range': [1.96, 10], 'color': "#27AE60"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 1.96
                }
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 2: LO ---
    with tab2:
        st.markdown("### Explore Learning Outcomes & Power Analysis")
        item_sel_lo = st.selectbox("Select an item to pre-fill actual values:", df_lo['Question'].tolist(), key="lo_sel")
        actual_lo = df_lo[df_lo['Question'] == item_sel_lo].iloc[0]

        if actual_lo['Dev_from_Mean'] > 0 and actual_lo['p_value'] > 0.05:
            st.warning(f"**Insight:** This item ('{item_sel_lo}') has a positive deviation but didn't reach significance. With $N={int(actual_lo['N'])}$, the standard error is too large. If the same deviation were observed with a larger sample size, it might become significant.")
        elif actual_lo['Dev_from_Mean'] < 0:
            st.warning(f"**Insight:** This item ('{item_sel_lo}') scores below the Siena average. The negative deviation reflects curriculum alignment, not teaching quality.")

        colA, colB, colC = st.columns(3)
        c_mean_lo = colA.number_input("Your Mean", value=float(actual_lo['Avg']), step=0.05, key="lo_mean")
        siena_mean_lo = colB.number_input("Siena Mean", value=float(actual_lo['Siena_Avg']), step=0.05, key="lo_siena")
        sd_lo = colC.number_input("Standard Deviation", value=float(actual_lo['SD']), step=0.05, key="lo_sd")
        
        n_slider = st.slider("What if N were larger? (Power Analysis)", min_value=50, max_value=500, value=int(actual_lo['N']), step=10)

        # Computations
        dev_lo = c_mean_lo - siena_mean_lo
        se_lo = sd_lo / np.sqrt(n_slider)
        t_stat_lo = dev_lo / se_lo
        df_stat_lo = n_slider - 1
        p_val_lo = 2 * stats.t.sf(abs(t_stat_lo), df_stat_lo)

        st.markdown("---")
        st.markdown(f"**At N = {n_slider}:**")
        st.markdown(f"**Deviation** = {c_mean_lo:.2f} - {siena_mean_lo:.2f} = `{dev_lo:+.2f}`")
        st.markdown(f"**t-statistic** = `{t_stat_lo:.2f}`")
        st.markdown(f"**p-value** = `{p_val_lo:.4f}`")

        is_sig_lo = p_val_lo < 0.05
        if is_sig_lo:
            st.success(f"Significant at p < 0.05? **YES**")
        else:
            st.error(f"Significant at p < 0.05? **NO**")

        # Plotly curve for Power Analysis
        n_range = np.arange(50, 501, 10)
        p_vals = []
        for n_test in n_range:
            se_test = sd_lo / np.sqrt(n_test)
            t_test = dev_lo / se_test
            p_test = 2 * stats.t.sf(abs(t_test), n_test - 1)
            p_vals.append(p_test)
            
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=n_range, y=p_vals, mode='lines', name='p-value'))
        fig_power.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Significance Threshold (p=0.05)")
        fig_power.add_trace(go.Scatter(x=[n_slider], y=[p_val_lo], mode='markers', marker=dict(size=12, color='black'), name=f'Current N={n_slider}'))
        
        fig_power.update_layout(
            title="How Sample Size (N) Affects Significance",
            xaxis_title="Sample Size (N)",
            yaxis_title="p-value (log scale)",
            yaxis_type="log",
            height=400
        )
        st.plotly_chart(fig_power, use_container_width=True)

# --- PAGE 5: METHODOLOGY ---
elif page == "Methodology":
    st.title("Methodology")

    st.markdown("### Research Questions")
    st.markdown("**RQ1:** Does this instructor’s teaching effectiveness, as measured by student evaluation scores on core instructor items, differ significantly from the Siena University average?")
    st.markdown("**RQ2:** Beyond statistical significance, what is the practical magnitude of student satisfaction—specifically, what proportion of the student population reports favorable experiences?")
    st.markdown("**RQ3:** Do student-reported learning gains in discipline-relevant outcomes exceed institutional norms?")

    st.markdown("### Hypothesis Testing")
    st.markdown("For each evaluation item, the one-sample t-test evaluates:")
    st.markdown("- **H₀ (Null):** The instructor’s mean score equals the Siena University population mean ($\mu_{instructor} = \mu_{Siena}$). Any observed deviation is attributable to sampling variability.")
    st.markdown("- **H₁ (Alternative):** The instructor’s mean score differs from the Siena University mean ($\mu_{instructor} \\neq \mu_{Siena}$). The observed deviation is too large to be explained by chance.")

    st.markdown("### Z-Score vs. T-Statistic")
    st.markdown("The SmartEvals system provides a Z-Score, but a one-sample T-test is strictly more appropriate because it systematically accounts for our sample standard deviation and sample size ($N$).")
    
    st.latex(r"t = \frac{\bar{X} - \mu_S}{s / \sqrt{n}}")

    st.markdown("### Limitations and Notes")
    st.markdown("- **Statistical Power:** Core items have $N \\approx 392$, yielding high power. Learning outcomes have $N \\approx 135$, meaning moderate differences may not reach statistical significance.")
    st.markdown("- **Top-Box Top-Heavy:** Because mean scores are naturally right-skewed (most scores fall in 6 or 7), Top-Box analysis (Agree + Strongly Agree) complements standard mean testing to demonstrate practical significance.")
