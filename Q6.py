import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Is there a relationship between peer influence and exam scores?


def jonckheere_terpstra_test(groups):
    '''
    Jonckheere–Terpstra test for ordered alternatives.
    groups: list of arrays, ordered from lowest to highest category.
    '''

    groups = [np.array(g) for g in groups]

    JT = 0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g_i = groups[i]
            g_j = groups[j]

            for x in g_i:
                JT += np.sum(g_j > x) + 0.5 * np.sum(g_j == x)
    
    n = np.array([len(g) for g in groups])
    N = np.sum(n)

    mean_JT = (N**2 - sum(n**2)) / 4

    var_JT = (
        (N*(N-1)*(2*N+5) - np.sum(n*(n-1)*(2*n+5))) / 72
    )

    z = (JT - mean_JT) / np.sqrt(var_JT)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return JT, z, p_value


df['Peer_Influence'] = df['Peer_Influence'].astype('category')
df['Peer_Influence'] = df['Peer_Influence'].cat.set_categories(
    ['Negative', 'Neutral', 'Positive'],
    ordered=True
)

groups = [
    df[df['Peer_Influence'] == level]['Exam_Score']
    for level in df['Peer_Influence'].cat.categories
]

JT, z, p_value = jonckheere_terpstra_test(groups)

print(f"JT statistic = {JT:.3f}")
print(f"z-value = {z:.3f}")
print(f"p-value = {p_value:.6f}")



'''
To examine whether exam performance increases with more positive peer influence, a Jonckheere–Terpstra 
trend test was conducted with Peer Influence treated as an ordered factor (Negative < Neutral < Positive).
The test indicated a moderately positive monotonic trend in exam scores across the levels 
of peer influence (JT = 7,770,018.5, z = 9.00, p < 0.001).

This result suggests that students who report more positive peer influence tend to achieve higher exam scores, 
and that this improvement follows a consistent increasing pattern from negative to neutral to positive peer 
environments.
'''

# Visualising with a ridgeline plot

plt.figure(figsize=(10, 6))
sns.pointplot(
    data=df,
    x='Peer_Influence',
    y='Exam_Score',
    capsize=.15,
    ci=95,
    color='green'
)
plt.title('Mean Exam Score (95% CI) by Peer Influence Level')
plt.xlabel('Peer Influence')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.show()


custom_colors = ["#46017F", "#22A5C9", "#86C611"]

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x='Peer_Influence',
    y='Exam_Score',
    showcaps=True,
    boxprops={'facecolor': 'none'},
    showfliers=False,
    color='black'
)
sns.stripplot(
    data=df,
    x='Peer_Influence',
    y='Exam_Score',
    alpha=0.35,
    jitter=0.25,
    palette=custom_colors
)
plt.title('Exam Scores by Peer Influence Level')
plt.xlabel('Peer Influence')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.show()