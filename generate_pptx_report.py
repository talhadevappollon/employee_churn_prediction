"""
PowerPoint Report Generator for Statistical Analysis
====================================================
This script generates a professional PPTX report with the statistical findings
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import pandas as pd
from pathlib import Path
from datetime import datetime


class PPTXReportGenerator:
    """Generates a professional PowerPoint report for statistical analysis"""

    def __init__(self):
        """Initialize the presentation"""
        self.prs = Presentation()
        self.prs.slide_width = Inches(10)
        self.prs.slide_height = Inches(7.5)

        # Define color scheme
        self.colors = {
            'primary': RGBColor(0, 51, 102),      # Dark blue
            'secondary': RGBColor(0, 112, 192),   # Light blue
            'success': RGBColor(0, 153, 76),      # Green
            'warning': RGBColor(255, 153, 0),     # Orange
            'danger': RGBColor(192, 0, 0),        # Red
            'text': RGBColor(51, 51, 51),         # Dark gray
            'light': RGBColor(242, 242, 242)      # Light gray
        }

        # Load analysis results
        self.results_dir = Path('analysis_results')
        self.summary_df = pd.read_csv(self.results_dir / 'statistical_summary.csv')

    def add_title_slide(self):
        """Add title slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])  # Blank layout

        # Add background color
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = self.colors['primary']

        # Add title
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(1))
        title_frame = title_box.text_frame
        title_frame.text = "Statistical Validation of HR Churn Predictors"
        title_para = title_frame.paragraphs[0]
        title_para.alignment = PP_ALIGN.CENTER
        title_para.font.size = Pt(44)
        title_para.font.bold = True
        title_para.font.color.rgb = RGBColor(255, 255, 255)

        # Add subtitle
        subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.2), Inches(9), Inches(0.6))
        subtitle_frame = subtitle_box.text_frame
        subtitle_frame.text = "T-Test & Chi-Square Analysis Report"
        subtitle_para = subtitle_frame.paragraphs[0]
        subtitle_para.alignment = PP_ALIGN.CENTER
        subtitle_para.font.size = Pt(28)
        subtitle_para.font.color.rgb = RGBColor(200, 200, 200)

        # Add date and author
        info_box = slide.shapes.add_textbox(Inches(0.5), Inches(6), Inches(9), Inches(0.5))
        info_frame = info_box.text_frame
        info_frame.text = f"Statistical Analysis Report | {datetime.now().strftime('%B %d, %Y')}"
        info_para = info_frame.paragraphs[0]
        info_para.alignment = PP_ALIGN.CENTER
        info_para.font.size = Pt(16)
        info_para.font.color.rgb = RGBColor(180, 180, 180)

    def add_agenda_slide(self):
        """Add agenda slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])

        # Title
        self._add_slide_title(slide, "Agenda")

        # Content
        content_box = slide.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(7), Inches(4.5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True

        agenda_items = [
            "1. Executive Summary",
            "2. Methodology & Approach",
            "3. Feature Analysis: TotalWorkingYears",
            "4. Feature Analysis: NumCompaniesWorked",
            "5. Feature Analysis: OverTime",
            "6. Feature Analysis: WorkLifeBalance",
            "7. Comparative Results Summary",
            "8. Statistical Confidence Assessment",
            "9. Recommendations & Conclusions"
        ]

        for i, item in enumerate(agenda_items):
            p = text_frame.add_paragraph() if i > 0 else text_frame.paragraphs[0]
            p.text = item
            p.font.size = Pt(20)
            p.font.color.rgb = self.colors['text']
            p.space_after = Pt(12)
            p.level = 0

    def add_executive_summary_slide(self):
        """Add executive summary"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Executive Summary")

        # Key findings box
        findings_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(1.5))
        text_frame = findings_box.text_frame
        text_frame.word_wrap = True

        significant_count = (self.summary_df['Significant'] == '✓').sum()
        total_features = len(self.summary_df)

        summary_text = f"""OBJECTIVE: Validate the statistical significance of {total_features} features recommended by the Data Science team for predicting employee churn.

RESULT: {significant_count} out of {total_features} features demonstrated statistically significant associations with employee attrition (p < 0.05)."""

        p = text_frame.paragraphs[0]
        p.text = summary_text
        p.font.size = Pt(16)
        p.font.color.rgb = self.colors['text']
        p.line_spacing = 1.3

        # Add findings table
        y_pos = 3.5
        self._add_summary_table(slide, Inches(1), Inches(y_pos), Inches(8), Inches(2.5))

        # Overall assessment
        assessment_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.2), Inches(8.4), Inches(0.8))
        assessment_frame = assessment_box.text_frame

        if significant_count == total_features:
            assessment = "✓ VALIDATION COMPLETE: All recommended features are statistically reliable for churn prediction."
            color = self.colors['success']
        elif significant_count >= 3:
            assessment = "✓ MOSTLY VALIDATED: Majority of features are reliable. Focus on significant predictors."
            color = self.colors['success']
        elif significant_count >= 2:
            assessment = "⚠ PARTIALLY VALIDATED: Some features reliable. Recommend additional feature exploration."
            color = self.colors['warning']
        else:
            assessment = "✗ REQUIRES REVISION: Insufficient validation. Alternative features recommended."
            color = self.colors['danger']

        p = assessment_frame.paragraphs[0]
        p.text = assessment
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color

    def _add_summary_table(self, slide, left, top, width, height):
        """Add summary table to slide"""
        rows = len(self.summary_df) + 1
        cols = 4

        table = slide.shapes.add_table(rows, cols, left, top, width, height).table

        # Set column widths
        table.columns[0].width = Inches(2.5)
        table.columns[1].width = Inches(2)
        table.columns[2].width = Inches(1.5)
        table.columns[3].width = Inches(2)

        # Header row
        headers = ['Feature', 'Test Type', 'Significant', 'Effect Size']
        for col_idx, header in enumerate(headers):
            cell = table.cell(0, col_idx)
            cell.text = header
            cell.fill.solid()
            cell.fill.fore_color.rgb = self.colors['primary']
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.font.bold = True
            paragraph.font.size = Pt(12)
            paragraph.font.color.rgb = RGBColor(255, 255, 255)
            paragraph.alignment = PP_ALIGN.CENTER

        # Data rows
        for row_idx, row_data in self.summary_df.iterrows():
            cells_data = [
                row_data['Feature'],
                row_data['Test'],
                row_data['Significant'],
                row_data['Effect Size'].title()
            ]

            for col_idx, cell_data in enumerate(cells_data):
                cell = table.cell(row_idx + 1, col_idx)
                cell.text = str(cell_data)
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.font.size = Pt(11)
                paragraph.alignment = PP_ALIGN.CENTER

                # Color code significance
                if col_idx == 2:
                    if cell_data == '✓':
                        paragraph.font.color.rgb = self.colors['success']
                        paragraph.font.bold = True
                    else:
                        paragraph.font.color.rgb = self.colors['danger']
                        paragraph.font.bold = True

    def add_methodology_slide(self):
        """Add methodology slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Methodology & Statistical Approach")

        # Content
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True

        methodology_text = """STATISTICAL TESTS APPLIED:

1. Independent T-Test (Continuous Variables)
   • Used for: TotalWorkingYears, NumCompaniesWorked
   • Tests: Difference in means between churners vs non-churners
   • Significance Level: α = 0.05
   • Effect Size Measure: Cohen's d

2. Chi-Square Test of Independence (Categorical Variables)
   • Used for: OverTime, WorkLifeBalance
   • Tests: Association between feature categories and attrition
   • Significance Level: α = 0.05
   • Effect Size Measure: Cramér's V

VALIDATION CRITERIA:
✓ p-value < 0.05: Feature is statistically significant
✓ Effect Size: Quantifies practical significance (negligible/small/medium/large)
✓ Confidence Levels: 95%, 99%, 99.9%"""

        p = text_frame.paragraphs[0]
        p.text = methodology_text
        p.font.size = Pt(14)
        p.font.color.rgb = self.colors['text']
        p.line_spacing = 1.4

    def add_feature_slide(self, feature_name, row_data):
        """Add detailed feature analysis slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, f"Feature Analysis: {feature_name}")

        # Add chart image
        img_path = self.results_dir / f"{feature_name}_analysis.png"
        if img_path.exists():
            slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.6), width=Inches(9))

        # Add interpretation box at bottom
        interp_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(0.8))
        interp_frame = interp_box.text_frame

        test_type = row_data['Test']
        p_value = row_data['p-value']
        significant = row_data['Significant'] == '✓'
        effect_size = row_data['Effect Size']

        if significant:
            interpretation = f"✓ SIGNIFICANT (p={p_value:.4f}, {effect_size.title()} effect) - {row_data['Recommendation']}"
            color = self.colors['success']
        else:
            interpretation = f"✗ NOT SIGNIFICANT (p={p_value:.4f}) - {row_data['Recommendation']}"
            color = self.colors['danger']

        p = interp_frame.paragraphs[0]
        p.text = interpretation
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p.alignment = PP_ALIGN.CENTER

    def add_comparison_slide(self):
        """Add comparison summary slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Comparative Results Summary")

        # Add summary comparison image
        img_path = self.results_dir / "summary_comparison.png"
        if img_path.exists():
            slide.shapes.add_picture(str(img_path), Inches(0.5), Inches(1.6), width=Inches(9))

    def add_confidence_slide(self):
        """Add statistical confidence assessment"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Statistical Confidence Assessment")

        # Content
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True

        confidence_text = "CONFIDENCE LEVELS BY FEATURE:\n\n"

        for _, row in self.summary_df.iterrows():
            feature = row['Feature']
            confidence = row['Confidence']
            significant = row['Significant']

            icon = "✓" if significant == '✓' else "✗"
            confidence_text += f"{icon} {feature}\n   Confidence: {confidence}\n\n"

        confidence_text += """\nINTERPRETATION GUIDE:
• 99.9% (p < 0.001): Extremely strong evidence - Highest confidence
• 99% (p < 0.01): Very strong evidence - Very high confidence
• 95% (p < 0.05): Strong evidence - High confidence
• Below 95%: Insufficient evidence - Not reliable for prediction"""

        p = text_frame.paragraphs[0]
        p.text = confidence_text
        p.font.size = Pt(13)
        p.font.color.rgb = self.colors['text']
        p.line_spacing = 1.3

    def add_recommendations_slide(self):
        """Add recommendations slide"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Recommendations & Conclusions")

        # Content
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True

        significant_features = self.summary_df[self.summary_df['Significant'] == '✓']['Feature'].tolist()
        non_significant_features = self.summary_df[self.summary_df['Significant'] != '✓']['Feature'].tolist()

        recommendations = "KEY FINDINGS:\n\n"

        if significant_features:
            recommendations += f"✓ STATISTICALLY VALIDATED FEATURES ({len(significant_features)}):\n"
            for feat in significant_features:
                recommendations += f"   • {feat}\n"
            recommendations += "\n"

        if non_significant_features:
            recommendations += f"✗ NON-VALIDATED FEATURES ({len(non_significant_features)}):\n"
            for feat in non_significant_features:
                recommendations += f"   • {feat}\n"
            recommendations += "\n"

        recommendations += """\nACTIONABLE RECOMMENDATIONS:

1. PRIORITIZE validated features in your predictive model
2. WEIGHT features based on effect size magnitude
3. CONSIDER removing non-significant features to reduce noise
4. MONITOR model performance with validated feature set
5. EXPLORE additional features if validation rate is low

CONCLUSION:
The statistical analysis provides objective validation of feature importance.
Use these insights to build a more reliable and interpretable churn prediction model."""

        p = text_frame.paragraphs[0]
        p.text = recommendations
        p.font.size = Pt(13)
        p.font.color.rgb = self.colors['text']
        p.line_spacing = 1.3

    def add_appendix_slide(self):
        """Add technical appendix"""
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._add_slide_title(slide, "Technical Appendix")

        # Content
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(5))
        text_frame = content_box.text_frame
        text_frame.word_wrap = True

        appendix_text = """EFFECT SIZE INTERPRETATIONS:

Cohen's d (T-Test):
• < 0.2: Negligible effect
• 0.2 - 0.5: Small effect
• 0.5 - 0.8: Medium effect
• > 0.8: Large effect

Cramér's V (Chi-Square):
• < 0.1: Negligible association
• 0.1 - 0.3: Small association
• 0.3 - 0.5: Medium association
• > 0.5: Large association

ASSUMPTIONS & LIMITATIONS:
• Sample size: N = 1471 employees
• Missing data handled via exclusion
• Independence assumption verified
• Both parametric and non-parametric tests conducted

TOOLS & LIBRARIES:
• Python 3.x with SciPy for statistical tests
• Pandas for data manipulation
• Matplotlib/Seaborn for visualizations"""

        p = text_frame.paragraphs[0]
        p.text = appendix_text
        p.font.size = Pt(12)
        p.font.color.rgb = self.colors['text']
        p.line_spacing = 1.3

    def _add_slide_title(self, slide, title_text):
        """Add title to slide"""
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
        title_frame = title_box.text_frame
        title_frame.text = title_text

        # Title formatting
        p = title_frame.paragraphs[0]
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = self.colors['primary']

        # Add horizontal line
        line = slide.shapes.add_shape(
            1,  # Line shape
            Inches(0.5), Inches(1.2), Inches(9), Inches(0)
        )
        line.line.color.rgb = self.colors['secondary']
        line.line.width = Pt(3)

    def generate_report(self, output_filename='HR_Churn_Statistical_Analysis_Report.pptx'):
        """Generate complete PowerPoint report"""
        print("\n" + "="*70)
        print("GENERATING POWERPOINT REPORT")
        print("="*70)

        # Add all slides
        print("\n📄 Adding Title Slide...")
        self.add_title_slide()

        print("📄 Adding Agenda...")
        self.add_agenda_slide()

        print("📄 Adding Executive Summary...")
        self.add_executive_summary_slide()

        print("📄 Adding Methodology...")
        self.add_methodology_slide()

        # Add feature-specific slides
        print("📄 Adding Feature Analysis Slides...")
        for _, row in self.summary_df.iterrows():
            feature_name = row['Feature']
            print(f"   • {feature_name}")
            self.add_feature_slide(feature_name, row)

        print("📄 Adding Comparison Summary...")
        self.add_comparison_slide()

        print("📄 Adding Confidence Assessment...")
        self.add_confidence_slide()

        print("📄 Adding Recommendations...")
        self.add_recommendations_slide()

        print("📄 Adding Technical Appendix...")
        self.add_appendix_slide()

        # Save presentation
        self.prs.save(output_filename)

        print("\n" + "="*70)
        print(f"✓ PowerPoint report generated successfully!")
        print(f"  File: {output_filename}")
        print(f"  Total slides: {len(self.prs.slides)}")
        print("="*70)


def main():
    """Main execution"""
    generator = PPTXReportGenerator()
    generator.generate_report()


if __name__ == "__main__":
    main()
