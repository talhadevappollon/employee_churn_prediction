"""
Statistical Analysis of HR Churn Prediction Features
=====================================================
This script performs T-Tests and Chi-Square tests to validate the relationship
between key features and employee attrition.

Features analyzed:
- TotalWorkingYears (Continuous - T-Test)
- OverTime (Categorical - Chi-Square)
- WorkLifeBalance (Ordinal - Chi-Square)
- NumCompaniesWorked (Discrete - T-Test)

Target: Attrition (Yes/No)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ChurnFeatureAnalyzer:
    """Analyzes the statistical significance of features related to churn"""

    def __init__(self, data_path):
        """Load and prepare the data"""
        self.df = pd.read_csv(data_path, encoding='utf-8-sig')
        self.results = {}
        self.output_dir = Path('analysis_results')
        self.output_dir.mkdir(exist_ok=True)

        print("="*70)
        print("HR CHURN PREDICTION - STATISTICAL FEATURE VALIDATION")
        print("="*70)
        print(f"\nDataset loaded: {len(self.df)} records")
        print(f"Attrition distribution:")
        print(self.df['Attrition'].value_counts())
        print(f"\nAttrition rate: {(self.df['Attrition']=='Yes').sum()/len(self.df)*100:.2f}%")
        print("\n" + "="*70)

    def analyze_continuous_feature(self, feature_name):
        """
        Perform T-Test for continuous variables
        Tests if there's a significant difference in means between churners and non-churners
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {feature_name} (Continuous Variable)")
        print(f"{'='*70}")

        # Separate data by attrition status
        churners = self.df[self.df['Attrition'] == 'Yes'][feature_name]
        non_churners = self.df[self.df['Attrition'] == 'No'][feature_name]

        # Remove any missing values
        churners = churners.dropna()
        non_churners = non_churners.dropna()

        # Descriptive statistics
        print(f"\nDescriptive Statistics:")
        print(f"  Churners (n={len(churners)}):")
        print(f"    Mean: {churners.mean():.2f}")
        print(f"    Median: {churners.median():.2f}")
        print(f"    Std Dev: {churners.std():.2f}")
        print(f"    Range: [{churners.min():.2f}, {churners.max():.2f}]")

        print(f"\n  Non-Churners (n={len(non_churners)}):")
        print(f"    Mean: {non_churners.mean():.2f}")
        print(f"    Median: {non_churners.median():.2f}")
        print(f"    Std Dev: {non_churners.std():.2f}")
        print(f"    Range: [{non_churners.min():.2f}, {non_churners.max():.2f}]")

        # Check normality (Shapiro-Wilk test for samples < 5000)
        _, p_norm_churn = stats.shapiro(churners.sample(min(5000, len(churners))))
        _, p_norm_no_churn = stats.shapiro(non_churners.sample(min(5000, len(non_churners))))

        print(f"\n  Normality Tests (Shapiro-Wilk):")
        print(f"    Churners p-value: {p_norm_churn:.4f} {'(Normal)' if p_norm_churn > 0.05 else '(Non-normal)'}")
        print(f"    Non-Churners p-value: {p_norm_no_churn:.4f} {'(Normal)' if p_norm_no_churn > 0.05 else '(Non-normal)'}")

        # Perform Independent T-Test
        t_statistic, p_value_ttest = ttest_ind(churners, non_churners)

        # Also perform Mann-Whitney U test (non-parametric alternative)
        u_statistic, p_value_mannwhitney = mannwhitneyu(churners, non_churners, alternative='two-sided')

        # Calculate effect size (Cohen's d)
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

        print(f"\n  Statistical Tests:")
        print(f"    Independent T-Test:")
        print(f"      t-statistic: {t_statistic:.4f}")
        print(f"      p-value: {p_value_ttest:.6f}")
        print(f"      Result: {'SIGNIFICANT ✓' if p_value_ttest < 0.05 else 'NOT SIGNIFICANT ✗'}")

        print(f"\n    Mann-Whitney U Test (non-parametric):")
        print(f"      U-statistic: {u_statistic:.4f}")
        print(f"      p-value: {p_value_mannwhitney:.6f}")
        print(f"      Result: {'SIGNIFICANT ✓' if p_value_mannwhitney < 0.05 else 'NOT SIGNIFICANT ✗'}")

        print(f"\n  Effect Size:")
        print(f"    Cohen's d: {cohens_d:.4f}")
        print(f"    Interpretation: {effect_interpretation.upper()}")

        # Confidence level
        if p_value_ttest < 0.001:
            confidence = "99.9% (Extremely High)"
        elif p_value_ttest < 0.01:
            confidence = "99% (Very High)"
        elif p_value_ttest < 0.05:
            confidence = "95% (High)"
        else:
            confidence = "Below 95% (Low - Not Reliable)"

        print(f"\n  CONFIDENCE LEVEL: {confidence}")

        # Store results
        self.results[feature_name] = {
            'type': 'continuous',
            'churners_mean': churners.mean(),
            'non_churners_mean': non_churners.mean(),
            'churners_std': churners.std(),
            'non_churners_std': non_churners.std(),
            't_statistic': t_statistic,
            'p_value_ttest': p_value_ttest,
            'p_value_mannwhitney': p_value_mannwhitney,
            'cohens_d': cohens_d,
            'effect_size': effect_interpretation,
            'significant': p_value_ttest < 0.05,
            'confidence': confidence
        }

        # Create visualization
        self._plot_continuous_distribution(feature_name, churners, non_churners)

        return self.results[feature_name]

    def analyze_categorical_feature(self, feature_name):
        """
        Perform Chi-Square Test for categorical variables
        Tests if there's a significant association between the feature and attrition
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {feature_name} (Categorical Variable)")
        print(f"{'='*70}")

        # Create contingency table
        contingency_table = pd.crosstab(self.df[feature_name], self.df['Attrition'])
        print(f"\nContingency Table:")
        print(contingency_table)

        # Calculate proportions
        print(f"\nAttrition Rates by Category:")
        proportions = pd.crosstab(self.df[feature_name], self.df['Attrition'], normalize='index') * 100
        print(proportions.round(2))

        # Perform Chi-Square Test
        chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

        # Calculate Cramér's V (effect size for chi-square)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        # Effect size interpretation for Cramér's V
        if cramers_v < 0.1:
            effect_interpretation = "negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "small"
        elif cramers_v < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        print(f"\n  Chi-Square Test Results:")
        print(f"    Chi-square statistic: {chi2:.4f}")
        print(f"    Degrees of freedom: {dof}")
        print(f"    p-value: {p_value:.6f}")
        print(f"    Result: {'SIGNIFICANT ✓' if p_value < 0.05 else 'NOT SIGNIFICANT ✗'}")

        print(f"\n  Effect Size:")
        print(f"    Cramér's V: {cramers_v:.4f}")
        print(f"    Interpretation: {effect_interpretation.upper()}")

        # Confidence level
        if p_value < 0.001:
            confidence = "99.9% (Extremely High)"
        elif p_value < 0.01:
            confidence = "99% (Very High)"
        elif p_value < 0.05:
            confidence = "95% (High)"
        else:
            confidence = "Below 95% (Low - Not Reliable)"

        print(f"\n  CONFIDENCE LEVEL: {confidence}")

        # Expected frequencies
        print(f"\n  Expected Frequencies:")
        print(pd.DataFrame(expected_freq,
                          index=contingency_table.index,
                          columns=contingency_table.columns).round(2))

        # Store results
        self.results[feature_name] = {
            'type': 'categorical',
            'contingency_table': contingency_table,
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': cramers_v,
            'effect_size': effect_interpretation,
            'significant': p_value < 0.05,
            'confidence': confidence,
            'attrition_rates': proportions['Yes'].to_dict()
        }

        # Create visualization
        self._plot_categorical_distribution(feature_name, contingency_table, proportions)

        return self.results[feature_name]

    def _plot_continuous_distribution(self, feature_name, churners, non_churners):
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
        result = self.results[feature_name]
        summary_text = f"""
        STATISTICAL SUMMARY

        Churners Mean: {result['churners_mean']:.2f} ± {result['churners_std']:.2f}
        Non-Churners Mean: {result['non_churners_mean']:.2f} ± {result['non_churners_std']:.2f}

        T-Test p-value: {result['p_value_ttest']:.6f}
        Significance: {'YES ✓' if result['significant'] else 'NO ✗'}

        Cohen's d: {result['cohens_d']:.4f}
        Effect Size: {result['effect_size'].upper()}

        Confidence: {result['confidence']}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{feature_name}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n  📊 Visualization saved: {feature_name}_analysis.png")
        plt.close()

    def _plot_categorical_distribution(self, feature_name, contingency_table, proportions):
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

        # 3. Heatmap of contingency table
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[1, 0],
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='black')
        axes[1, 0].set_title('Contingency Table Heatmap', fontweight='bold')
        axes[1, 0].set_xlabel('Attrition', fontweight='bold')
        axes[1, 0].set_ylabel(feature_name, fontweight='bold')

        # 4. Statistical summary
        result = self.results[feature_name]
        summary_text = f"""
        STATISTICAL SUMMARY

        Chi-Square: {result['chi2']:.4f}
        p-value: {result['p_value']:.6f}
        Degrees of Freedom: {result['dof']}

        Significance: {'YES ✓' if result['significant'] else 'NO ✗'}

        Cramér's V: {result['cramers_v']:.4f}
        Effect Size: {result['effect_size'].upper()}

        Confidence: {result['confidence']}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{feature_name}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n  📊 Visualization saved: {feature_name}_analysis.png")
        plt.close()

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE SUMMARY REPORT")
        print(f"{'='*70}")

        print("\n┌" + "─"*68 + "┐")
        print("│" + " "*20 + "FEATURE VALIDATION SUMMARY" + " "*22 + "│")
        print("└" + "─"*68 + "┘")

        summary_data = []

        for feature, result in self.results.items():
            print(f"\n{feature}:")
            print(f"  Type: {result['type'].capitalize()}")
            print(f"  Statistical Test: {'T-Test' if result['type'] == 'continuous' else 'Chi-Square'}")

            if result['type'] == 'continuous':
                print(f"  p-value: {result['p_value_ttest']:.6f}")
                print(f"  Effect Size (Cohen's d): {result['cohens_d']:.4f} ({result['effect_size']})")
            else:
                print(f"  p-value: {result['p_value']:.6f}")
                print(f"  Effect Size (Cramér's V): {result['cramers_v']:.4f} ({result['effect_size']})")

            print(f"  Significant: {'✓ YES' if result['significant'] else '✗ NO'}")
            print(f"  Confidence: {result['confidence']}")

            recommendation = self._get_recommendation(result)
            print(f"  Recommendation: {recommendation}")

            summary_data.append({
                'Feature': feature,
                'Test': 'T-Test' if result['type'] == 'continuous' else 'Chi-Square',
                'p-value': result['p_value_ttest'] if result['type'] == 'continuous' else result['p_value'],
                'Significant': '✓' if result['significant'] else '✗',
                'Effect Size': result['effect_size'],
                'Confidence': result['confidence'],
                'Recommendation': recommendation
            })

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)

        # Save summary to CSV
        summary_df.to_csv(self.output_dir / 'statistical_summary.csv', index=False)
        print(f"\n  📄 Summary report saved: statistical_summary.csv")

        # Create summary visualization
        self._plot_summary_chart(summary_df)

        # Overall conclusion
        print(f"\n{'='*70}")
        print("FINAL CONCLUSION")
        print(f"{'='*70}")

        significant_features = [f for f, r in self.results.items() if r['significant']]
        non_significant_features = [f for f, r in self.results.items() if not r['significant']]

        print(f"\n✓ Statistically Significant Features ({len(significant_features)}):")
        for feat in significant_features:
            print(f"  • {feat} - {self.results[feat]['confidence']}")

        if non_significant_features:
            print(f"\n✗ Non-Significant Features ({len(non_significant_features)}):")
            for feat in non_significant_features:
                print(f"  • {feat}")

        print(f"\n{'='*70}")
        print("RECOMMENDATIONS FOR DATA SCIENTIST")
        print(f"{'='*70}")

        if len(significant_features) == 4:
            print("\n✓ ALL FEATURES ARE STATISTICALLY VALIDATED")
            print("  The Data Scientist's recommendations are RELIABLE.")
            print("  All features show significant association with employee churn.")
        elif len(significant_features) >= 3:
            print(f"\n✓ MOSTLY VALIDATED ({len(significant_features)}/4 features)")
            print("  The Data Scientist's recommendations are LARGELY RELIABLE.")
            print(f"  Consider focusing on: {', '.join(significant_features)}")
        elif len(significant_features) >= 2:
            print(f"\n⚠ PARTIALLY VALIDATED ({len(significant_features)}/4 features)")
            print("  Some features are reliable, but caution is advised.")
            print(f"  Prioritize: {', '.join(significant_features)}")
        else:
            print(f"\n✗ INSUFFICIENT VALIDATION ({len(significant_features)}/4 features)")
            print("  The Data Scientist's recommendations need revision.")
            print("  Consider exploring alternative features.")

        return summary_df

    def _get_recommendation(self, result):
        """Generate recommendation based on test results"""
        if result['significant']:
            if result['effect_size'] in ['large', 'medium']:
                return "HIGHLY RECOMMENDED - Strong predictor"
            else:
                return "RECOMMENDED - Valid predictor"
        else:
            return "NOT RECOMMENDED - Weak/No association"

    def _plot_summary_chart(self, summary_df):
        """Create a summary visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Statistical Validation Summary - All Features', fontsize=16, fontweight='bold')

        # 1. P-values comparison
        colors = ['green' if sig == '✓' else 'red' for sig in summary_df['Significant']]
        axes[0].barh(summary_df['Feature'], -np.log10(summary_df['p-value']), color=colors, edgecolor='black')
        axes[0].axvline(x=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='Significance Threshold (p=0.05)')
        axes[0].axvline(x=-np.log10(0.01), color='darkblue', linestyle='--', linewidth=2, label='High Significance (p=0.01)')
        axes[0].set_xlabel('-log10(p-value)', fontweight='bold')
        axes[0].set_ylabel('Feature', fontweight='bold')
        axes[0].set_title('Statistical Significance', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')

        # 2. Effect sizes
        effect_mapping = {'negligible': 1, 'small': 2, 'medium': 3, 'large': 4}
        effect_values = [effect_mapping[es] for es in summary_df['Effect Size']]
        colors2 = ['lightcoral' if v == 1 else 'gold' if v == 2 else 'lightgreen' if v == 3 else 'darkgreen'
                  for v in effect_values]
        axes[1].barh(summary_df['Feature'], effect_values, color=colors2, edgecolor='black')
        axes[1].set_xlabel('Effect Size', fontweight='bold')
        axes[1].set_ylabel('Feature', fontweight='bold')
        axes[1].set_title('Effect Size Magnitude', fontweight='bold')
        axes[1].set_xticks([1, 2, 3, 4])
        axes[1].set_xticklabels(['Negligible', 'Small', 'Medium', 'Large'])
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  📊 Summary visualization saved: summary_comparison.png")
        plt.close()


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = ChurnFeatureAnalyzer('Dept_Pred_HR_Attrition.csv')

    # Analyze continuous features (T-Test)
    print("\n\n" + "█"*70)
    print("PART 1: CONTINUOUS FEATURES ANALYSIS (T-TESTS)")
    print("█"*70)

    analyzer.analyze_continuous_feature('TotalWorkingYears')
    analyzer.analyze_continuous_feature('NumCompaniesWorked')

    # Analyze categorical features (Chi-Square)
    print("\n\n" + "█"*70)
    print("PART 2: CATEGORICAL FEATURES ANALYSIS (CHI-SQUARE TESTS)")
    print("█"*70)

    analyzer.analyze_categorical_feature('OverTime')
    analyzer.analyze_categorical_feature('WorkLifeBalance')

    # Generate comprehensive summary
    print("\n\n" + "█"*70)
    print("PART 3: COMPREHENSIVE SUMMARY")
    print("█"*70)

    summary_df = analyzer.generate_summary_report()

    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. TotalWorkingYears_analysis.png")
    print("  2. NumCompaniesWorked_analysis.png")
    print("  3. OverTime_analysis.png")
    print("  4. WorkLifeBalance_analysis.png")
    print("  5. summary_comparison.png")
    print("  6. statistical_summary.csv")
    print("\nAll files saved in: analysis_results/")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
