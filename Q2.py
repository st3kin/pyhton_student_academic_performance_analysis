import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, spearmanr

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Is there a relationship between hours studied and exam scores?

stat1, p1 = shapiro(df['Hours_Studied'])
stat2, p2 = shapiro(df['Exam_Score'])

print(f"Shapiro Hours_Studied p-value: {p1:.6f}")
print(f"Shapiro Exam_Score p-value: {p2:.6f}")

'''
The scores are not normally distributed, so I'll use Spearman instead of Pearson.
'''

hours = df['Hours_Studied']
scores = df['Exam_Score']

spearman_rho, spearman_p = spearmanr(hours, scores)

print(f"Spearman rho: {spearman_rho:.4f}, p-value: {spearman_p:.6f}")

'''
Spearman's correlation results indicate that students who spend more hours studying tend 
to achieve higher exam scores. The association is moderately strong, suggesting that study 
time is an important predictor of academic performance.
'''

# Visualising with scatter plot + regression line

plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x='Hours_Studied',
    y='Exam_Score',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'red'}
)
plt.title('Relationship Between Hours Studied and Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()
