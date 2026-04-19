# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier




#Loading Dataset

df_raw = pd.read_csv("data/kaggle_survey_2022_responses.csv", low_memory=False)

#Data Preprocessing
df = df_raw.iloc[1:].copy()
df.reset_index(drop=True, inplace=True)
selected_features = ['Q2','Q4','Q8','Q23','Q11','Q16','Q29']

eda_df = df[selected_features].copy()

# Removing rows with missing values
eda_df = eda_df.dropna(subset = ['Q29'])
eda_df.isna().sum()

# Filling missing values in Q16 with "Not Answered"
eda_df['Q16'] = eda_df['Q16'].fillna("Not Answered")
eda_df.isna().sum()


q12_cols = [col for col in df.columns if col.startswith('Q12_')]
eda_df["num_of_programming_languages"] = df[q12_cols].notnull().sum(axis=1)


age_map = {
    '18-21': 19.5,
    '22-24': 23,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-69': 64.5,
    '70+': 75
}

eda_df['age_numeric'] = eda_df['Q2'].map(age_map)

coding_xp_map ={
    'I have never written code':0,
    '< 1 years':0.5,
    '1-3 years':2,
    '3-5 years':4,
    '5-10 years':7.5,
    '10-20 years':15,
    '20+ years':25
}
eda_df['coding_xp_numeric'] = eda_df['Q11'].map(coding_xp_map)


ml_exp_map = {
    'I do not use machine learning methods': 0,
    'Under 1 year': 0.5,
    '1-2 years': 1.5,
    '2-3 years': 2.5,
    '3-4 years': 3.5,
    '4-5 years': 4.5,
    '5-10 years':7.5,
    '10-20 years': 15,
    'Not Answered': -1
}

eda_df['ml_exp_numeric'] = eda_df['Q16'].map(ml_exp_map)

income_map = {
    '$0-999': 500,
    '1,000-1,999': 1500,
    '2,000-2,999': 2500,
    '3,000-3,999': 3500,
    '4,000-4,999': 4500,
    '5,000-7,499': 6250,
    '7,500-9,999': 8750,
    '10,000-14,999': 12500,
    '15,000-19,999': 17500,
    '20,000-24,999': 22500,
    '25,000-29,999': 27500,
    '30,000-39,999': 35000,
    '40,000-49,999': 45000,
    '50,000-59,999': 55000,
    '60,000-69,999': 65000,
    '70,000-79,999': 75000,
    '80,000-89,999': 85000,
    '90,000-99,999': 95000,
    '100,000-124,999': 112500,
    '125,000-149,999': 137500,
    '150,000-199,999': 175000,
    '200,000-249,999': 225000,
    '250,000-299,999': 275000,
    '300,000-499,999': 400000,
    '$500,000-999,999': 750000,
    '>$1,000,000' : 1500000
}

eda_df['income_numeric'] = eda_df['Q29'].map(income_map)

# Creating binary target variable for income
features = [
    'age_numeric',
    'coding_xp_numeric',
    'ml_exp_numeric',
    'num_of_programming_languages',
]


eda_df['income_target'] = eda_df['income_numeric'].apply(
    lambda x: '< $50k' if x < 55000 else '≥ $50k'
)

X = eda_df[features]
y = eda_df['income_target'].map({'< $50k': 0, '≥ $50k': 1})




X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=67
)


rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print(classification_report(y_test, rf_pred))


ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Baseline Random Forest Classification Confusion Matrix")
plt.show()


