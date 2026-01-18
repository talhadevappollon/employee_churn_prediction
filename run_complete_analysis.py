"""
Master Script: Run Complete Statistical Analysis and Generate Report
"""

import subprocess
import sys

print("\n" + "="*70)
print("RUNNING STATISTICAL ANALYSIS PIPELINE")
print("="*70)

print("\n[STEP 1/2] Running Statistical Analysis...")
subprocess.run([sys.executable, 'statistical_analysis.py'], check=True)

print("\n[STEP 2/2] Generating PowerPoint Report...")
subprocess.run([sys.executable, 'generate_pptx_report.py'], check=True)

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print("="*70)
print("\nDeliverables:")
print("  1. Statistical analysis code: statistical_analysis.py")
print("  2. Analysis outputs: analysis_results/ folder")
print("  3. PowerPoint report: HR_Churn_Statistical_Analysis_Report.pptx")
print("\n" + "="*70)
