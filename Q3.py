import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp


# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Does level of access to resources relate to exam score?

df['Access_to_Resources'] = df['Access_to_Resources'].astype('category')
df['Access_to_Resources'] = df['Access_to_Resources'].cat.set_categories(
    ['Low', 'Medium', 'High'],
    ordered=True
)

col = 'Access_to_Resources'
groups = [df[df[col] == level]['Exam_Score'] for level in df[col].cat.categories]

H, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis H = {H:.3f}, p = {p:.6f}")

# Conducting a post-hoc pairwise comparison using Dunn's test with Bonferroni correction

posthoc = sp.posthoc_dunn(
    df,
    val_col='Exam_Score',
    group_col='Access_to_Resources',
    p_adjust='holm'
)

print(posthoc)

'''
A Kruskal–Wallis test revealed a statistically significant difference in exam performance across levels of access to academic 
resources (H = 231.678, p < 0.001). Post-hoc Dunn comparisons with Holm correction showed that all three groups differed 
significantly from one another (all p < 0.001), with exam scores increasing progressively from Low to Medium to High access 
levels. This result indicates a strong and monotonic association between resource availability and academic achievement in 
this dataset.
'''

# Visualising with a violin and point plot

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df,
    x='Access_to_Resources',
    y='Exam_Score',
    inner='quartile',
    cut=0,
    palette='viridis'
)
plt.title('Access to Resources vs Exam Score')
plt.xlabel('Level of Access to Resources')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.pointplot(
    data=df,
    x='Access_to_Resources',
    y='Exam_Score',
    capsize=.15,
    ci=99,
    color='purple'
)
plt.title('Average Exam Score (CI 99%) by Level of Access to Resources')
plt.show()

