"""
Title: Basics of Panel Data Analysis and Visualization in Python
Description: End-to-end script covering data loading, multi-dimensional 
             visualizations, diagnostics, and static panel data models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. DATA LOADING & BASIC MANAGEMENT
# ==========================================
print('--- 1. Data Loading & Management ---')

# Example 1: Loading built-in Seaborn datasets for immediate practice
tips = sns.load_dataset('tips')
fmri = sns.load_dataset('fmri')

print("Tips Dataset Head:\n", tips.head(5))
print('\nTips Columns:', tips.columns.tolist())
print('\nDescriptive Statistics:\n', tips.describe())

# Data Transformation Examples:
# Generating a squared term and log transformation
tips['total_bill_sq'] = tips['total_bill'] ** 2
tips['log_tip'] = np.log(tips['tip'] + 1)


# ==========================================
# 2. DATA VISUALIZATION WORKFLOW
# ==========================================
print('\n--- 2. Generating Visualizations ---')

# Set aesthetic style
sns.set_theme(style='whitegrid')

# A. Multidimensional Scatter Plot (4D representation: X, Y, Hue, Style, Size)
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='time',
    style='sex',
    size='size',
    sizes=(20, 200),
)
plt.title('Multidimensional Scatter Plot (Bill vs. Tip)')
plt.xlabel('Total Bill Amount')
plt.ylabel('Tip Amount')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scatter_plot_4d.png', dpi=300)  # High-res export
plt.show()

# B. Box Plot
plt.figure(figsize=(6, 4))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Total Bill by Day and Gender')
plt.show()

# C. Violin Plot (Distribution + Box summary)
plt.figure(figsize=(6, 4))
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
plt.title('Distribution of Total Bill (Violin Plot)')
plt.show()

# D. Time Series Line Plot
plt.figure(figsize=(10, 4))
sns.lineplot(data=fmri, x='timepoint', y='signal', hue='region')
plt.title('Time Series Signal Analysis')
plt.show()


# ==========================================
# 3. STATIC PANEL DATA ANALYSIS & REGRESSIONS
# ==========================================
print('\n--- 3. Panel Data Regression Setup ---')

# Simulating a panel structure for demonstration purposes 
# (Replace 'panel_data.csv' with your actual dataset file path)
np.random.seed(42)
n_entities = 10
n_years = 5
entity_ids = [f'Country_{i}' for i in range(1, n_entities + 1)]
years = list(range(2015, 2015 + n_years))

panel_data_list = []
for ent in entity_ids:
  for yr in years:
    panel_data_list.append({
        'id_code': ent,
        'year': yr,
        'salary': np.random.normal(50000, 5000) + (yr - 2015) * 1000,
        'age': np.random.randint(25, 60),
        'tenure': np.random.randint(1, 15),
    })

data = pd.DataFrame(panel_data_list)

# Configure Panel Index (equivalent to xtset in Stata)
# drop=False keeps the actual columns available in the dataframe
data = data.set_index(['id_code', 'year'], drop=False)
print(data.head())

# A. Pooled OLS Regression
import statsmodels.formula.api as smf

pooled_reg = smf.ols('salary ~ age + tenure', data=data).fit()
print('\n--- Pooled OLS Results ---')
print(pooled_reg.summary())

# B. Heteroscedasticity Check (Breusch-Pagan Test)
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(pooled_reg.resid, pooled_reg.model.exog)
print(f'\nBreusch-Pagan Test P-Value: {bp_test[1]:.4f}')

# C. Fixed Effects & Random Effects Models
from linearmodels.panel import PanelOLS, RandomEffects

# Fixed Effects Model
fe_model = PanelOLS.from_formula(
    'salary ~ age + tenure + EntityEffects', data=data
)
fe_results = fe_model.fit()
print('\n--- Fixed Effects Model Results ---')
print(fe_results)

# Random Effects Model
re_model = RandomEffects.from_formula('salary ~ age + tenure', data=data)
re_results = re_model.fit()
print('\n--- Random Effects Model Results ---')
print(re_results)

print(
    '\nScript execution complete! You can replace the simulated panel data'
    ' with your own .csv, .dta, or .sav file imports.'
)
