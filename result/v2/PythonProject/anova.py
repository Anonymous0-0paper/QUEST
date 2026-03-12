import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Excel file
file_path = '../results-excel/result-Alibaba-5.xlsx'  # Replace with your actual file path
sheet_name = 'Makespan'

# Read the data
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Clean and prepare data for analysis
# Extract data from rows 2-17 (algorithms with their load values)
data = df.iloc[1:17].copy()  # Skip the header row

# Since we have limited samples per group, let's use an approach that works better with small samples
# First, let's analyze descriptive statistics

print("Descriptive Statistics by Algorithm:")
algorithm_stats = data.groupby('Algorithm')['Average'].agg(['mean', 'std', 'min', 'max']).reset_index()
print(algorithm_stats)

# Create a boxplot to visualize the data
plt.figure(figsize=(12, 6))
sns.boxplot(x='Algorithm', y='Average', data=data)
plt.title('Performance Comparison by Algorithm')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('algorithm_comparison.png')

# Instead of ANOVA, we can use Kruskal-Wallis test which works better with small samples
# (non-parametric alternative to one-way ANOVA)
algorithms = data['Algorithm'].unique()
kw_groups = [data[data['Algorithm'] == algo]['Average'].values for algo in algorithms]

# Filter out empty groups
kw_groups = [group for group in kw_groups if len(group) > 0]

# Run Kruskal-Wallis test
if all(len(group) > 0 for group in kw_groups):
    kw_statistic, kw_pvalue = stats.kruskal(*kw_groups)

    print("\nKruskal-Wallis Test Results (by Algorithm):")
    print(f"Statistic: {kw_statistic:.4f}")
    print(f"p-value: {kw_pvalue:.4f}")

    if kw_pvalue < 0.05:
        print("There are significant differences between algorithm performances (p < 0.05)")
    else:
        print("No significant differences detected between algorithm performances (p >= 0.05)")
else:
    print("\nCannot run Kruskal-Wallis test: One or more groups has no valid data")

# Let's also look at the effect of Load
plt.figure(figsize=(10, 6))
sns.boxplot(x='Load', y='Average', data=data)
plt.title('Performance Comparison by Load')
plt.ylabel('Average Value')
plt.tight_layout()
plt.savefig('load_comparison.png')

# Visualize interaction between Algorithm and Load
plt.figure(figsize=(12, 6))
sns.barplot(x='Algorithm', y='Average', hue='Load', data=data)
plt.title('Performance Comparison by Algorithm and Load')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.legend(title='Load')
plt.tight_layout()
plt.savefig('algorithm_load_comparison.png')