import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Are exam performance levels independent of parental involvement? (Chi-square)


print(df['Exam_Score_Level'].value_counts())

df['Exam_Score_Level_Chi2'] = pd.qcut(
    df['Exam_Score'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

contingency_table = pd.crosstab(
    df['Exam_Score_Level_Chi2'],
    df['Parental_Involvement']
)

row_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p:.6f}")

# Visualisations

plt.figure(figsize=(10, 6))
sns.heatmap(
    row_pct,
    annot=True,
    fmt='.2f',
    cmap='Blues'
)
plt.title('Proportion Heatmap: Exam Score Level vs Parental Involvement')
plt.xlabel('Parental Involvement')
plt.ylabel('Exam Score Level')
plt.show()



'''
Although the chi-square test showed a statistically significant association between parental involvement and exam performance level 
(p < 0.001), the heatmap reveals that the relationship is not linear. Instead of a simple progression from Low > Medium > High parental 
involvement, the distribution varies across categories. This indicates that exam performance depends on parental involvement in a more 
complex way. Medium involvement is the most common category overall, which explains why raw counts are highest in that group across all 
score levels. Thus, although the association is statistically significant, it is not strictly monotonic.
'''
