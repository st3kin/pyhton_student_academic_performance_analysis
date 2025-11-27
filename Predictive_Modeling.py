import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

# Defining the groups

numeric_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']

ordinal_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality',
                    'Parental_Education_Level', 'Distance_from_Home', 'Peer_Influence']

binary_features = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities']

nominal_features = ['School_Type', 'Gender']

target = 'Exam_Score'


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


ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('ord', ordinal_encoder, ordinal_features),
        ('bin', 'passthrough', binary_features),
        ('nom', OneHotEncoder(drop='first'), nominal_features)
    ]
)

# Train-test split

X = df[numeric_features + ordinal_features + binary_features + nominal_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Building the Random Forest pipeline

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Evaluating model performance

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Extracting feature importance

feature_names = (
    numeric_features +
    ordinal_features +
    binary_features +
    list(model.named_steps['preprocess'].transformers_[3][1].get_feature_names_out(nominal_features))
)

importances = model.named_steps['rf'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(15))

'''
A Random Forest regression model was trained to predict students’ exam performance using academic habits, demographic variables, 
socioeconomic indicators, and lifestyle factors. The model demonstrated strong predictive accuracy (R² = 0.672, MAE = 1.07, RMSE = 2.15), 
indicating that approximately 67% of the variance in exam scores could be explained by the included predictors. 

Several socioeconomic and contextual variables showed statistically significant associations with exam performance in nonparametric tests. 
However, when included alongside stronger academic engagement measures in a predictive model, these factors contributed relatively little 
additional explanatory power. This suggests that while such variables are related to achievement, their influence is largely indirect and 
mediated through behaviours such as attendance and study effort.

The feature importance analysis indicates that attendance, hours studied, and previous academic performance are the dominant predictors of 
exam score. Other factors, including family income, parental education level, and peer influence, add only modest predictive value once 
these primary learning-related variables are accounted for. Based on these results we can conclude that a variable may be statistically 
significant in isolation, while still being a weak predictor in a multivariate.
'''

# Visualising

top_n = 12

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df.head(top_n)['Feature'][::-1],
    importance_df.head(top_n)['Importance'][::-1],
    color='#4DBBD5'
)
plt.title('Top Predictors of Exam Score')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()