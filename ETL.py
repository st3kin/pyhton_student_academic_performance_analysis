import pandas as pd
import numpy as np


# Importing the data

df = pd.read_csv('CSV_files/StudentPerformanceFactors.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Missing Values:")
print(df.isnull().sum())

df.info()

# Converting columns to correct datatypes

binary_categories = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']
ordinal_categories = {
    'Parental_Involvement': ['Low', 'Medium', 'High'], 
    'Access_to_Resources': ['Low', 'Medium', 'High'], 
    'Motivation_Level': ['Low', 'Medium', 'High'], 
    'Family_Income': ['Low', 'Medium', 'High'], 
    'Teacher_Quality': ['Low', 'Medium', 'High'], 
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Near', 'Moderate', 'Far'], 
    'Peer_Influence': ['Negative', 'Neutral', 'Positive']
}
nominal_categories = ['School_Type', 'Gender']


binary_map = {'Yes': 1, 'No': 0}

for col in binary_categories:
    df[col] = df[col].map(binary_map).astype('bool')

for col in nominal_categories:
    df[col] = df[col].astype('category')

for col, order in ordinal_categories.items():
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.set_categories(order, ordered=True)

print(df.dtypes)
print(df['Parental_Involvement'].cat.categories)
print(df['Parental_Involvement'].cat.ordered)

# New helper columns

def categorise_score(score):

    if score < 60:
        return 'Low'
    
    elif score <=80:
        return 'Medium'
    
    else:
        return 'High'

df['Exam_Score_Level'] = df['Exam_Score'].apply(categorise_score)
df['Exam_Score_Level'] = df['Exam_Score_Level'].astype('category').cat.set_categories(
    ['Low', 'Medium', 'High'], ordered=True
)

print(df['Exam_Score_Level'].value_counts())

# Imputing missing values with mode

'''
Since the variables “Teacher Quality,” “Parental Education Level,” and “Distance from Home” contain less than 1.5% 
null values each, the missing values were imputed using the mode. This approach maintains statistical power, 
preserves sample size for machine learning models, and avoids unnecessary row deletion.
'''

cols_to_impute = [
    'Teacher_Quality',
    'Parental_Education_Level',
    'Distance_from_Home'
]

for col in cols_to_impute:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)

# Exporting cleaned dataset

df.to_csv('CSV_files/processed_df.csv', index=False, encoding='utf-8')

