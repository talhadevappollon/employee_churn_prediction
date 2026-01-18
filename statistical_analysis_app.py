"""
Statistical Analysis App for HR Churn Prediction Features
==========================================================
Interactive Streamlit app for T-Test and Chi-Square analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="HR Churn Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1a237e !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(248, 249, 250, 0.1);
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: rgba(232, 245, 233, 0.1);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    /* Warning Card - Orange - Force white text in dark mode */
    .warning-card {
        background-color: rgba(255, 152, 0, 0.2) !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border-left: 5px solid #ff9800 !important;
    }
    .warning-card h4,
    .warning-card h3,
    .warning-card p,
    .warning-card strong {
        color: #ffffff !important;
    }
    /* Success Card - Green - Force white text in dark mode */
    .success-card {
        background-color: rgba(76, 175, 80, 0.2) !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border-left: 5px solid #4caf50 !important;
    }
    .success-card h4,
    .success-card h3,
    .success-card p,
    .success-card strong {
        color: #ffffff !important;
    }
    /* Danger Card - Red - Force white text in dark mode */
    .danger-card {
        background-color: rgba(244, 67, 54, 0.2) !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        border-left: 5px solid #f44336 !important;
    }
    .danger-card h4,
    .danger-card h3,
    .danger-card p,
    .danger-card strong {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Helper Functions
@st.cache_data
def load_data(uploaded_file=None):
    """Load the HR data"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    else:
        try:
            df = pd.read_csv('Dept_Pred_HR_Attrition.csv', encoding='utf-8-sig')
        except FileNotFoundError:
            return None
    return df

def perform_ttest(df, feature_name, target='Attrition'):
    """Perform T-Test for continuous variables"""
    churners = df[df[target] == 'Yes'][feature_name].dropna()
    non_churners = df[df[target] == 'No'][feature_name].dropna()

    # Descriptive statistics
    results = {
        'churners_mean': churners.mean(),
        'non_churners_mean': non_churners.mean(),
        'churners_median': churners.median(),
        'non_churners_median': non_churners.median(),
        'churners_std': churners.std(),
        'non_churners_std': non_churners.std(),
        'churners_n': len(churners),
        'non_churners_n': len(non_churners)
    }

    # T-Test
    t_statistic, p_value_ttest = ttest_ind(churners, non_churners)

    # Mann-Whitney U test (non-parametric)
    u_statistic, p_value_mannwhitney = mannwhitneyu(churners, non_churners, alternative='two-sided')

    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(churners)-1)*churners.std()**2 + (len(non_churners)-1)*non_churners.std()**2) /
                        (len(churners) + len(non_churners) - 2))
    cohens_d = (churners.mean() - non_churners.mean()) / pooled_std

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    results.update({
        't_statistic': t_statistic,
        'p_value_ttest': p_value_ttest,
        'p_value_mannwhitney': p_value_mannwhitney,
        'cohens_d': cohens_d,
        'effect_size': effect_interpretation,
        'significant': p_value_ttest < 0.05,
        'churners_data': churners,
        'non_churners_data': non_churners
    })

    return results

def perform_chi_square(df, feature_name, target='Attrition'):
    """Perform Chi-Square Test for categorical variables"""
    # Create contingency table
    contingency_table = pd.crosstab(df[feature_name], df[target])

    # Calculate proportions
    proportions = pd.crosstab(df[feature_name], df[target], normalize='index') * 100

    # Chi-Square Test
    chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

    # Cramér's V (effect size)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    # Effect size interpretation
    if cramers_v < 0.1:
        effect_interpretation = "negligible"
    elif cramers_v < 0.3:
        effect_interpretation = "small"
    elif cramers_v < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    results = {
        'contingency_table': contingency_table,
        'proportions': proportions,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected_freq': expected_freq,
        'cramers_v': cramers_v,
        'effect_size': effect_interpretation,
        'significant': p_value < 0.05,
        'attrition_rates': proportions['Yes'].to_dict() if 'Yes' in proportions.columns else {}
    }

    return results

def get_confidence_level(p_value):
    """Get confidence level from p-value"""
    if p_value < 0.001:
        return "99.9% (Extremely High)", "success"
    elif p_value < 0.01:
        return "99% (Very High)", "success"
    elif p_value < 0.05:
        return "95% (High)", "info"
    else:
        return "Below 95% (Low - Not Reliable)", "error"

def plot_continuous_analysis(feature_name, churners, non_churners, results):
    """Create visualizations for continuous variables"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{feature_name} Analysis: Churners vs Non-Churners', fontsize=16, fontweight='bold')

    # 1. Histogram comparison
    axes[0, 0].hist(non_churners, alpha=0.6, label='No Attrition', bins=30, color='green', edgecolor='black')
    axes[0, 0].hist(churners, alpha=0.6, label='Attrition', bins=30, color='red', edgecolor='black')
    axes[0, 0].set_xlabel(feature_name, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Distribution Comparison', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    data_for_box = pd.DataFrame({
        feature_name: list(churners) + list(non_churners),
        'Attrition': ['Yes']*len(churners) + ['No']*len(non_churners)
    })
    sns.boxplot(data=data_for_box, x='Attrition', y=feature_name, ax=axes[0, 1], palette=['red', 'green'])
    axes[0, 1].set_title('Box Plot Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Attrition Status', fontweight='bold')
    axes[0, 1].set_ylabel(feature_name, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Violin plot
    sns.violinplot(data=data_for_box, x='Attrition', y=feature_name, ax=axes[1, 0], palette=['red', 'green'])
    axes[1, 0].set_title('Violin Plot (Distribution Shape)', fontweight='bold')
    axes[1, 0].set_xlabel('Attrition Status', fontweight='bold')
    axes[1, 0].set_ylabel(feature_name, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Statistical summary
    summary_text = f"""
STATISTICAL SUMMARY

Churners Mean: {results['churners_mean']:.2f} ± {results['churners_std']:.2f}
Non-Churners Mean: {results['non_churners_mean']:.2f} ± {results['non_churners_std']:.2f}

T-Test p-value: {results['p_value_ttest']:.6f}
Significance: {'YES ✓' if results['significant'] else 'NO ✗'}

Cohen's d: {results['cohens_d']:.4f}
Effect Size: {results['effect_size'].upper()}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig

def plot_categorical_analysis(feature_name, contingency_table, proportions, results):
    """Create visualizations for categorical variables"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{feature_name} Analysis: Association with Attrition', fontsize=16, fontweight='bold')

    # 1. Stacked bar chart (counts)
    contingency_table.plot(kind='bar', stacked=True, ax=axes[0, 0], color=['green', 'red'], edgecolor='black')
    axes[0, 0].set_title('Attrition Count by Category', fontweight='bold')
    axes[0, 0].set_xlabel(feature_name, fontweight='bold')
    axes[0, 0].set_ylabel('Count', fontweight='bold')
    axes[0, 0].legend(title='Attrition', labels=['No', 'Yes'])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Grouped bar chart (proportions)
    proportions.plot(kind='bar', ax=axes[0, 1], color=['green', 'red'], edgecolor='black')
    axes[0, 1].set_title('Attrition Rate by Category (%)', fontweight='bold')
    axes[0, 1].set_xlabel(feature_name, fontweight='bold')
    axes[0, 1].set_ylabel('Percentage', fontweight='bold')
    axes[0, 1].legend(title='Attrition', labels=['No', 'Yes'])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Heatmap
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[1, 0],
               cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='black')
    axes[1, 0].set_title('Contingency Table Heatmap', fontweight='bold')
    axes[1, 0].set_xlabel('Attrition', fontweight='bold')
    axes[1, 0].set_ylabel(feature_name, fontweight='bold')

    # 4. Statistical summary
    summary_text = f"""
STATISTICAL SUMMARY

Chi-Square: {results['chi2']:.4f}
p-value: {results['p_value']:.6f}
Degrees of Freedom: {results['dof']}

Significance: {'YES ✓' if results['significant'] else 'NO ✗'}

Cramér's V: {results['cramers_v']:.4f}
Effect Size: {results['effect_size'].upper()}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Statistical Validation of HR Churn Predictors</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: rgba(63, 81, 181, 0.15); border-radius: 10px; margin-bottom: 2rem; border: 1px solid rgba(63, 81, 181, 0.3);'>
        <h3 style='color: #ffffff; margin-bottom: 0.5rem;'>T-Test & Chi-Square Analysis</h3>
        <p style='color: #e0e0e0; margin: 0;'>Rigorous statistical validation of features recommended for employee churn prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📊 Statistical Analysis")
        st.markdown("---")
        st.title("⚙️ Settings")

        # Data upload option
        st.subheader("📁 Data Source")
        uploaded_file = st.file_uploader("Upload HR Data (CSV)", type=['csv'])

        if uploaded_file:
            st.success("✓ File uploaded successfully!")
        else:
            st.info("Using default dataset: Dept_Pred_HR_Attrition.csv")

        st.markdown("---")

        # Feature selection
        st.subheader("🎯 Features to Analyze")
        analyze_total_years = st.checkbox("TotalWorkingYears", value=True)
        analyze_num_companies = st.checkbox("NumCompaniesWorked", value=True)
        analyze_overtime = st.checkbox("OverTime", value=True)
        analyze_work_life = st.checkbox("WorkLifeBalance", value=True)

        st.markdown("---")

        # Settings
        st.subheader("📐 Statistical Settings")
        alpha_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        st.caption(f"Current threshold: {alpha_level} ({(1-alpha_level)*100:.0f}% confidence)")

        st.markdown("---")

        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        **Statistical Tests:**
        - **T-Test**: Continuous variables
        - **Chi-Square**: Categorical variables

        **Effect Sizes:**
        - **Cohen's d**: T-Test
        - **Cramér's V**: Chi-Square

        **Interpretation:**
        - p < 0.05: Significant
        - Effect sizes indicate practical importance
        """)

    # Load data
    df = load_data(uploaded_file)

    if df is None:
        st.error("❌ No data found! Please upload a CSV file or ensure 'Dept_Pred_HR_Attrition.csv' exists in the directory.")
        st.stop()

    # Data overview
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        churners = (df['Attrition'] == 'Yes').sum()
        st.metric("Churners", churners)
    with col3:
        non_churners = (df['Attrition'] == 'No').sum()
        st.metric("Non-Churners", non_churners)
    with col4:
        attrition_rate = (churners / len(df) * 100)
        st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    with st.expander("📊 View Data Sample"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Analysis Section
    st.header("🔬 Statistical Analysis")

    # Create tabs for different features
    tabs = []
    tab_names = []

    if analyze_total_years:
        tab_names.append("TotalWorkingYears")
    if analyze_num_companies:
        tab_names.append("NumCompaniesWorked")
    if analyze_overtime:
        tab_names.append("OverTime")
    if analyze_work_life:
        tab_names.append("WorkLifeBalance")

    if not tab_names:
        st.warning("⚠️ Please select at least one feature to analyze from the sidebar.")
        st.stop()

    tab_names.append("📊 Summary")
    tabs = st.tabs(tab_names)

    # Store results for summary
    all_results = []

    # TotalWorkingYears Analysis
    if analyze_total_years:
        with tabs[tab_names.index("TotalWorkingYears")]:
            st.subheader("📈 TotalWorkingYears Analysis")
            st.markdown("**Test Type:** Independent T-Test (Continuous Variable)")

            results = perform_ttest(df, 'TotalWorkingYears')

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Churners Mean", f"{results['churners_mean']:.2f} years")
            with col2:
                st.metric("Non-Churners Mean", f"{results['non_churners_mean']:.2f} years")
            with col3:
                st.metric("Difference", f"{abs(results['churners_mean'] - results['non_churners_mean']):.2f} years")
            with col4:
                confidence, status = get_confidence_level(results['p_value_ttest'])
                if status == "success":
                    st.success(f"✓ {confidence}")
                else:
                    st.error(f"✗ {confidence}")

            # Detailed statistics
            with st.expander("📊 Detailed Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Churners (n={}):**".format(results['churners_n']))
                    st.write(f"- Mean: {results['churners_mean']:.2f}")
                    st.write(f"- Median: {results['churners_median']:.2f}")
                    st.write(f"- Std Dev: {results['churners_std']:.2f}")
                with col2:
                    st.markdown("**Non-Churners (n={}):**".format(results['non_churners_n']))
                    st.write(f"- Mean: {results['non_churners_mean']:.2f}")
                    st.write(f"- Median: {results['non_churners_median']:.2f}")
                    st.write(f"- Std Dev: {results['non_churners_std']:.2f}")

                st.markdown("**Test Results:**")
                st.write(f"- T-statistic: {results['t_statistic']:.4f}")
                st.write(f"- p-value (T-Test): {results['p_value_ttest']:.6f}")
                st.write(f"- p-value (Mann-Whitney U): {results['p_value_mannwhitney']:.6f}")
                st.write(f"- Cohen's d: {results['cohens_d']:.4f}")
                st.write(f"- Effect Size: {results['effect_size'].title()}")

            # Visualization
            st.markdown("### 📊 Visual Analysis")
            fig = plot_continuous_analysis('TotalWorkingYears', results['churners_data'],
                                          results['non_churners_data'], results)
            st.pyplot(fig)

            # Interpretation
            st.markdown("### 💡 Interpretation")
            if results['significant']:
                st.markdown(f"""
                <div class="success-card">
                <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                <p>There is a <strong>significant difference</strong> in TotalWorkingYears between churners and non-churners
                (p = {results['p_value_ttest']:.6f}, α = {alpha_level}).</p>
                <p><strong>Effect Size:</strong> {results['effect_size'].title()} (Cohen's d = {results['cohens_d']:.4f})</p>
                <p><strong>Conclusion:</strong> This feature is a <strong>reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-card">
                <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                <p>There is <strong>no significant difference</strong> in TotalWorkingYears between churners and non-churners
                (p = {results['p_value_ttest']:.6f}, α = {alpha_level}).</p>
                <p><strong>Conclusion:</strong> This feature is <strong>not a reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)

            all_results.append({
                'Feature': 'TotalWorkingYears',
                'Test': 'T-Test',
                'p-value': results['p_value_ttest'],
                'Significant': '✓' if results['significant'] else '✗',
                'Effect Size': results['effect_size'].title(),
                'Effect Value': f"{results['cohens_d']:.4f}"
            })

    # NumCompaniesWorked Analysis
    if analyze_num_companies:
        with tabs[tab_names.index("NumCompaniesWorked")]:
            st.subheader("📈 NumCompaniesWorked Analysis")
            st.markdown("**Test Type:** Independent T-Test (Continuous Variable)")

            results = perform_ttest(df, 'NumCompaniesWorked')

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Churners Mean", f"{results['churners_mean']:.2f} companies")
            with col2:
                st.metric("Non-Churners Mean", f"{results['non_churners_mean']:.2f} companies")
            with col3:
                st.metric("Difference", f"{abs(results['churners_mean'] - results['non_churners_mean']):.2f} companies")
            with col4:
                confidence, status = get_confidence_level(results['p_value_ttest'])
                if status == "success":
                    st.success(f"✓ {confidence}")
                else:
                    st.error(f"✗ {confidence}")

            # Detailed statistics
            with st.expander("📊 Detailed Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Churners (n={}):**".format(results['churners_n']))
                    st.write(f"- Mean: {results['churners_mean']:.2f}")
                    st.write(f"- Median: {results['churners_median']:.2f}")
                    st.write(f"- Std Dev: {results['churners_std']:.2f}")
                with col2:
                    st.markdown("**Non-Churners (n={}):**".format(results['non_churners_n']))
                    st.write(f"- Mean: {results['non_churners_mean']:.2f}")
                    st.write(f"- Median: {results['non_churners_median']:.2f}")
                    st.write(f"- Std Dev: {results['non_churners_std']:.2f}")

                st.markdown("**Test Results:**")
                st.write(f"- T-statistic: {results['t_statistic']:.4f}")
                st.write(f"- p-value (T-Test): {results['p_value_ttest']:.6f}")
                st.write(f"- p-value (Mann-Whitney U): {results['p_value_mannwhitney']:.6f}")
                st.write(f"- Cohen's d: {results['cohens_d']:.4f}")
                st.write(f"- Effect Size: {results['effect_size'].title()}")

            # Visualization
            st.markdown("### 📊 Visual Analysis")
            fig = plot_continuous_analysis('NumCompaniesWorked', results['churners_data'],
                                          results['non_churners_data'], results)
            st.pyplot(fig)

            # Interpretation
            st.markdown("### 💡 Interpretation")
            if results['significant']:
                st.markdown(f"""
                <div class="success-card">
                <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                <p>There is a <strong>significant difference</strong> in NumCompaniesWorked between churners and non-churners
                (p = {results['p_value_ttest']:.6f}, α = {alpha_level}).</p>
                <p><strong>Effect Size:</strong> {results['effect_size'].title()} (Cohen's d = {results['cohens_d']:.4f})</p>
                <p><strong>Conclusion:</strong> This feature is a <strong>reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-card">
                <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                <p>There is <strong>no significant difference</strong> in NumCompaniesWorked between churners and non-churners
                (p = {results['p_value_ttest']:.6f}, α = {alpha_level}).</p>
                <p><strong>Conclusion:</strong> This feature is <strong>not a reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)

            all_results.append({
                'Feature': 'NumCompaniesWorked',
                'Test': 'T-Test',
                'p-value': results['p_value_ttest'],
                'Significant': '✓' if results['significant'] else '✗',
                'Effect Size': results['effect_size'].title(),
                'Effect Value': f"{results['cohens_d']:.4f}"
            })

    # OverTime Analysis
    if analyze_overtime:
        with tabs[tab_names.index("OverTime")]:
            st.subheader("📈 OverTime Analysis")
            st.markdown("**Test Type:** Chi-Square Test of Independence (Categorical Variable)")

            results = perform_chi_square(df, 'OverTime')

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            attrition_rates = results['attrition_rates']
            with col1:
                if 'No' in attrition_rates:
                    st.metric("No Overtime Attrition", f"{attrition_rates['No']:.2f}%")
            with col2:
                if 'Yes' in attrition_rates:
                    st.metric("With Overtime Attrition", f"{attrition_rates['Yes']:.2f}%")
            with col3:
                if 'Yes' in attrition_rates and 'No' in attrition_rates:
                    risk_multiplier = attrition_rates['Yes'] / attrition_rates['No']
                    st.metric("Risk Multiplier", f"{risk_multiplier:.2f}x")
            with col4:
                confidence, status = get_confidence_level(results['p_value'])
                if status == "success":
                    st.success(f"✓ {confidence}")
                else:
                    st.error(f"✗ {confidence}")

            # Contingency table
            with st.expander("📊 Contingency Table & Statistics"):
                st.markdown("**Observed Frequencies:**")
                st.dataframe(results['contingency_table'], use_container_width=True)

                st.markdown("**Attrition Rates by Category:**")
                st.dataframe(results['proportions'].round(2), use_container_width=True)

                st.markdown("**Test Results:**")
                st.write(f"- Chi-Square Statistic: {results['chi2']:.4f}")
                st.write(f"- p-value: {results['p_value']:.6f}")
                st.write(f"- Degrees of Freedom: {results['dof']}")
                st.write(f"- Cramér's V: {results['cramers_v']:.4f}")
                st.write(f"- Effect Size: {results['effect_size'].title()}")

            # Visualization
            st.markdown("### 📊 Visual Analysis")
            fig = plot_categorical_analysis('OverTime', results['contingency_table'],
                                           results['proportions'], results)
            st.pyplot(fig)

            # Interpretation
            st.markdown("### 💡 Interpretation")
            if results['significant']:
                st.markdown(f"""
                <div class="success-card">
                <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                <p>There is a <strong>significant association</strong> between OverTime and employee attrition
                (χ² = {results['chi2']:.4f}, p = {results['p_value']:.6f}, α = {alpha_level}).</p>
                <p><strong>Effect Size:</strong> {results['effect_size'].title()} (Cramér's V = {results['cramers_v']:.4f})</p>
                <p><strong>Conclusion:</strong> This feature is a <strong>reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-card">
                <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                <p>There is <strong>no significant association</strong> between OverTime and employee attrition
                (χ² = {results['chi2']:.4f}, p = {results['p_value']:.6f}, α = {alpha_level}).</p>
                <p><strong>Conclusion:</strong> This feature is <strong>not a reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)

            all_results.append({
                'Feature': 'OverTime',
                'Test': 'Chi-Square',
                'p-value': results['p_value'],
                'Significant': '✓' if results['significant'] else '✗',
                'Effect Size': results['effect_size'].title(),
                'Effect Value': f"{results['cramers_v']:.4f}"
            })

    # WorkLifeBalance Analysis
    if analyze_work_life:
        with tabs[tab_names.index("WorkLifeBalance")]:
            st.subheader("📈 WorkLifeBalance Analysis")
            st.markdown("**Test Type:** Chi-Square Test of Independence (Categorical Variable)")

            results = perform_chi_square(df, 'WorkLifeBalance')

            # Metrics - show attrition rates by rating
            attrition_rates = results['attrition_rates']
            cols = st.columns(len(attrition_rates) + 1)

            for idx, (rating, rate) in enumerate(sorted(attrition_rates.items())):
                with cols[idx]:
                    st.metric(f"Rating {rating}", f"{rate:.2f}%")

            with cols[-1]:
                confidence, status = get_confidence_level(results['p_value'])
                if status == "success":
                    st.success(f"✓ {confidence}")
                else:
                    st.error(f"✗ {confidence}")

            # Contingency table
            with st.expander("📊 Contingency Table & Statistics"):
                st.markdown("**Observed Frequencies:**")
                st.dataframe(results['contingency_table'], use_container_width=True)

                st.markdown("**Attrition Rates by Category:**")
                st.dataframe(results['proportions'].round(2), use_container_width=True)

                st.markdown("**Test Results:**")
                st.write(f"- Chi-Square Statistic: {results['chi2']:.4f}")
                st.write(f"- p-value: {results['p_value']:.6f}")
                st.write(f"- Degrees of Freedom: {results['dof']}")
                st.write(f"- Cramér's V: {results['cramers_v']:.4f}")
                st.write(f"- Effect Size: {results['effect_size'].title()}")

            # Visualization
            st.markdown("### 📊 Visual Analysis")
            fig = plot_categorical_analysis('WorkLifeBalance', results['contingency_table'],
                                           results['proportions'], results)
            st.pyplot(fig)

            # Interpretation
            st.markdown("### 💡 Interpretation")
            if results['significant']:
                st.markdown(f"""
                <div class="success-card">
                <h4>✅ STATISTICALLY SIGNIFICANT</h4>
                <p>There is a <strong>significant association</strong> between WorkLifeBalance and employee attrition
                (χ² = {results['chi2']:.4f}, p = {results['p_value']:.6f}, α = {alpha_level}).</p>
                <p><strong>Effect Size:</strong> {results['effect_size'].title()} (Cramér's V = {results['cramers_v']:.4f})</p>
                <p><strong>Conclusion:</strong> This feature is a <strong>reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-card">
                <h4>❌ NOT STATISTICALLY SIGNIFICANT</h4>
                <p>There is <strong>no significant association</strong> between WorkLifeBalance and employee attrition
                (χ² = {results['chi2']:.4f}, p = {results['p_value']:.6f}, α = {alpha_level}).</p>
                <p><strong>Conclusion:</strong> This feature is <strong>not a reliable predictor</strong> of employee churn.</p>
                </div>
                """, unsafe_allow_html=True)

            all_results.append({
                'Feature': 'WorkLifeBalance',
                'Test': 'Chi-Square',
                'p-value': results['p_value'],
                'Significant': '✓' if results['significant'] else '✗',
                'Effect Size': results['effect_size'].title(),
                'Effect Value': f"{results['cramers_v']:.4f}"
            })

    # Summary Tab
    with tabs[-1]:
        st.subheader("📊 Comprehensive Summary")

        if not all_results:
            st.warning("No features analyzed yet. Please select features from the sidebar.")
        else:
            # Summary dataframe
            summary_df = pd.DataFrame(all_results)

            # Overall metrics
            total_features = len(summary_df)
            significant_features = (summary_df['Significant'] == '✓').sum()
            validation_rate = (significant_features / total_features * 100)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Features Analyzed", total_features)
            with col2:
                st.metric("Validated Features", f"{significant_features}/{total_features}")
            with col3:
                if validation_rate >= 75:
                    st.success(f"✓ {validation_rate:.0f}% Validated")
                elif validation_rate >= 50:
                    st.warning(f"⚠ {validation_rate:.0f}% Validated")
                else:
                    st.error(f"✗ {validation_rate:.0f}% Validated")

            st.markdown("---")

            # Summary table
            st.markdown("### 📋 Results Table")

            # Style the dataframe
            def highlight_significance(row):
                if row['Significant'] == '✓':
                    return ['background-color: rgba(76, 175, 80, 0.25); color: #ffffff'] * len(row)
                else:
                    return ['background-color: rgba(244, 67, 54, 0.25); color: #ffffff'] * len(row)

            styled_df = summary_df.style.apply(highlight_significance, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Verdict
            st.markdown("### 🎯 Final Verdict")

            if validation_rate == 100:
                st.markdown("""
                <div class="success-card">
                <h3>✅ ALL FEATURES VALIDATED</h3>
                <p><strong>Conclusion:</strong> The Data Scientist's recommendations are <strong>COMPLETELY RELIABLE</strong>.</p>
                <p>All features show statistically significant associations with employee churn.</p>
                <p><strong>Recommendation:</strong> Proceed with confidence using all validated features in your predictive model.</p>
                </div>
                """, unsafe_allow_html=True)
            elif validation_rate >= 75:
                st.markdown(f"""
                <div class="success-card">
                <h3>✅ MOSTLY VALIDATED ({validation_rate:.0f}%)</h3>
                <p><strong>Conclusion:</strong> The Data Scientist's recommendations are <strong>LARGELY RELIABLE</strong>.</p>
                <p>Majority of features show statistically significant associations with employee churn.</p>
                <p><strong>Recommendation:</strong> Focus on the validated features: {', '.join(summary_df[summary_df['Significant']=='✓']['Feature'].tolist())}</p>
                </div>
                """, unsafe_allow_html=True)
            elif validation_rate >= 50:
                st.markdown(f"""
                <div class="warning-card">
                <h3>⚠️ PARTIALLY VALIDATED ({validation_rate:.0f}%)</h3>
                <p><strong>Conclusion:</strong> Some features are reliable, but caution is advised.</p>
                <p><strong>Validated Features:</strong> {', '.join(summary_df[summary_df['Significant']=='✓']['Feature'].tolist())}</p>
                <p><strong>Recommendation:</strong> Prioritize validated features and consider exploring alternative predictors.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-card">
                <h3>❌ INSUFFICIENT VALIDATION ({validation_rate:.0f}%)</h3>
                <p><strong>Conclusion:</strong> The Data Scientist's recommendations need significant revision.</p>
                <p><strong>Recommendation:</strong> Explore alternative features and reassess the predictive model approach.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Recommendations
            st.markdown("### 💡 Recommendations")

            validated = summary_df[summary_df['Significant'] == '✓']['Feature'].tolist()
            not_validated = summary_df[summary_df['Significant'] == '✗']['Feature'].tolist()

            col1, col2 = st.columns(2)

            with col1:
                if validated:
                    st.markdown("**✅ INCLUDE in Model:**")
                    for feat in validated:
                        st.markdown(f"- {feat}")
                else:
                    st.markdown("**✅ INCLUDE in Model:** None")

            with col2:
                if not_validated:
                    st.markdown("**❌ EXCLUDE from Model:**")
                    for feat in not_validated:
                        st.markdown(f"- {feat}")
                else:
                    st.markdown("**❌ EXCLUDE from Model:** None")

            # Download results
            st.markdown("---")
            st.markdown("### 💾 Download Results")

            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary as CSV",
                data=csv,
                file_name=f"statistical_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #b0b0b0; padding: 2rem;'>
        <p><strong>Statistical Analysis App</strong> | T-Test & Chi-Square Validation</p>
        <p>Developed for HR Churn Prediction Feature Validation</p>
        <p style='font-size: 0.8rem;'>© 2026 | Powered by Streamlit & SciPy</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
