import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Is there a relationship between attendance and exam scores?

attendance = df['Attendance']
scores = df['Exam_Score']

rho, p = spearmanr(attendance, scores)

print(f"Spearman rho: {rho:.4f}, p-value: {p:.6f}")

'''
The Spearman Correlation analysis revealed a strong positive correlation between attendance and exam scores 
(ρ = 0.6724, p < 0.001). This indicates that students who attend classes more frequently tend to achieve significantly 
higher exam scores. Among all behavioral variables examined so far, attendance demonstrates one of the strongest 
associations with academic success.
'''

# Visualising with hexbin plot

plt.figure(figsize=(10, 6))
plt.hexbin(df['Attendance'], df['Exam_Score'], gridsize=25, cmap='BuPu')
plt.colorbar(label='Count')
plt.xlabel('Attendance')
plt.ylabel('Exam Score')
plt.title('Attendance vs Exam Score')
plt.show()



