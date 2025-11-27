import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.api as sm

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Do physical activity and sleep influence academic success (i.e. more time spent exercising = higher exam scores / more sleep = higher exam scores)?

rho_pa, p_pa = spearmanr(df['Physical_Activity'], df['Exam_Score'])
print(f"Spearman rho (Physical Activity): {rho_pa:.4f}, p-value: {p_pa:.6f}")

rho_sl, p_sl = spearmanr(df['Sleep_Hours'], df['Exam_Score'])
print(f"Spearman rho (Sleep): {rho_sl:.4f}, p-value: {p_sl:.6f}")


'''
X = df[['Physical_Activity', 'Sleep_Hours']]
y = df['Exam_Score']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
'''

df['Sleep_Group'] = pd.qcut(
    df['Sleep_Hours'], q=4,
    labels=['Very Low','Low','Medium','High']
)

df['Activity_Group'] = pd.qcut(
    df['Physical_Activity'], q=4,
    labels=['Very Low','Low','Medium','High']
)

from scipy.stats import kruskal

sleep_groups = [df[df['Sleep_Group'] == level]['Exam_Score'] 
                for level in df['Sleep_Group'].cat.categories]

activity_groups = [df[df['Activity_Group'] == level]['Exam_Score'] 
                   for level in df['Activity_Group'].cat.categories]

sleep_H, sleep_p = kruskal(*sleep_groups)
activity_H, activity_p = kruskal(*activity_groups)

print(f"Sleep KW: H = {sleep_H:.3f}, p = {sleep_p:.6f}")
print(f"Activity KW: H = {activity_H:.3f}, p = {activity_p:.6f}")




'''
Spearman correlations were used to assess whether physical activity (hours per week) and sleep duration (avg hours per night) were associated with exam 
performance. Physical activity showed a very weak positive correlation with exam score (ρ = 0.029, p = 0.018), while sleep duration showed no meaningful 
relationship (ρ = −0.008, p = 0.535). Given the extremely small effect sizes, these results suggest that neither variable meaningfully predicts exam 
performance.

A multiple linear regression model using physical activity and sleep as predictors explained just 0.1% of the variance in exam scores (R² = 0.001). 
Physical activity had a statistically significant but practically negligible effect (β = 0.105, p = 0.024), while sleep had no significant effect 
(β = −0.045, p = 0.167). These findings indicate that students’ exercise and sleep habits, at least as measured in this dataset, do not substantially 
influence academic performance.
'''

# Visualising

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.regplot(
    data=df,
    x='Physical_Activity',
    y='Exam_Score',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'purple'}
)
plt.title('Exam Score vs Physical Activity')
plt.xlabel('Physical Activity (hours per week)')
plt.ylabel('Exam Score')

plt.subplot(1, 2, 2)
sns.regplot(
    data=df,
    x='Sleep_Hours',
    y='Exam_Score',
    scatter_kws={'alpha':0.3},
    line_kws={'color':'green'}
)
plt.title('Exam Score vs Sleep')
plt.xlabel('Sleep (avg hours per day)')
plt.ylabel('Exam Score')

plt.tight_layout()
plt.show()


