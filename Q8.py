import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from scipy.stats import chi2_contingency

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Is there a relationship between family income and exam scores?

df['Family_Income'] = df['Family_Income'].astype('category')
df['Family_Income'] = df['Family_Income'].cat.set_categories(
    ['Low', 'Medium', 'High'],
    ordered=True
)

col = 'Family_Income'
groups = [df[df[col] == level]['Exam_Score'] for level in df[col].cat.categories]

H, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H = {H:.3f}, p = {p:.6f}")

posthoc = sp.posthoc_dunn(
    df,
    val_col='Exam_Score',
    group_col='Family_Income',
    p_adjust='holm'
)

print(posthoc)

'''
A Kruskal–Wallis test indicated that exam performance differed significantly across family income levels (H = 63.80, p < 0.001).
Post-hoc Dunn tests with Holm correction revealed significant differences between all income groups: Low vs Medium (p < 0.001), 
Low vs High (p < 0.001), and Medium vs High (p < 0.001). These results demonstrate a consistent gradient in which students from 
higher-income families achieve higher exam scores, suggesting a clear socioeconomic influence on academic performance in this 
dataset.
'''

# Are there more (perceived) high-quality teachers in private schools or public schools?

contingency = pd.crosstab(df['School_Type'], df['Teacher_Quality'])
print(contingency)

chi2, chi_p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {chi_p:.6f}")

prop = contingency.div(contingency.sum(axis=1), axis=0)
print(prop)

'''
Proportional comparison of teacher quality between school types showed nearly identical distributions across Private and Public 
schools (High: 29.9% vs 29.3%, Medium: 60.5% vs 60.6%, Low: 9.7% vs 10.1%). Combined with a non-significant chi-square test 
(χ²(2) = 0.400, p = 0.819), these findings indicate that teacher quality does not differ meaningfully between school types in 
this dataset. Thus, private schools do not have a higher proportion of high-quality teachers than public schools.
'''

# Visualising with a 100% stacked bar chart

prop.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    color=["#4C297C", "#0591CE", "#40A03B"]
)
plt.title('Proportion of Teacher Quality by School Type')
plt.xlabel('School Type')
plt.ylabel('Teacher Quality')
plt.legend(title='Teacher Quality', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()