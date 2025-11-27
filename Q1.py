import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp


# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Does exam performance differ by students' parental education level?

df['Parental_Education_Level'] = df['Parental_Education_Level'].astype('category')
df['Parental_Education_Level'] = df['Parental_Education_Level'].cat.set_categories(
    ['High School', 'College', 'Postgraduate'],
    ordered=True
)

col = 'Parental_Education_Level'
groups = [df[df[col] == level]['Exam_Score'] for level in df[col].cat.categories]


print('Normality Tests (Shapiro-Wilk):')
for level in df[col].cat.categories:
    stat, p = stats.shapiro(df[df[col] == level]['Exam_Score'])
    print(f'{level}: p-value = {p:.4f}')

levene_stat, levene_p = stats.levene(*groups)
print(f"Levene's Test for Equal Variances: p = {levene_p:.4f}")

'''
Shapiro-Wilk tests for normality indicated that exam scores were not normally distributed within any parental 
education level group (all p-values were below 0.001). Since the test is highly sensitive for large samples, 
these significant results are expected and do not necessarily reflect severe deviations from normality. Regardless, 
non-parametric methods are more appropriate.

Therefore, I decided to use the Kruskal–Wallis H test.
'''

h_stat, p_val = stats.kruskal(*groups)
print(f"Kruskal-Wallis H = {h_stat:.3f}, p = {p_val:.5f}")

'''
The test indicated a statistically significant difference between the groups, H(2) = 93.24, p < 0.001.
This suggests that students’ exam performance varies depending on their parents' education level.
Because the overall test was significant, pairwise post-hoc comparisons are needed to determine which specific 
groups differed.
'''

# Dunn's test with Bonferroni correction for post-hoc pairwise comparisons

posthoc = sp.posthoc_dunn(
    df,
    val_col='Exam_Score',
    group_col='Parental_Education_Level',
    p_adjust='holm'
)

print(posthoc)

'''
               High School       College  Postgraduate
High School   1.000000e+00  1.748788e-05  3.038909e-21
College       1.748788e-05  1.000000e+00  1.617534e-07
Postgraduate  3.038909e-21  1.617534e-07  1.000000e+00

A Kruskal–Wallis test showed a statistically significant difference in exam performance across parental education 
levels (H = 93.241, p < 0.001). Post-hoc Dunn tests with Holm correction indicated that all three groups differed 
significantly from one another, with students whose parents held postgraduate degrees achieving the highest exam 
scores, followed by those with college-educated parents, and finally those whose parents completed high school. 
These findings suggest a clear and graded association between parental education level and academic performance.
'''

# Visualising with box and point plots

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x='Parental_Education_Level',
    y='Exam_Score',
    palette='viridis'
)
plt.title('Exam Score Distribution by Parental Education Level')
plt.xlabel('Parental Education Level')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(
    data=df,
    x='Parental_Education_Level',
    y='Exam_Score',
    capsize=.15,
    ci=95,
    color='black'
)
plt.title('Mean Exam Score (95% CI) by Parental Education Level')
plt.show()

