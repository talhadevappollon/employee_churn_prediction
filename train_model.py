"""
Train and save the churn prediction model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load data
df = pd.read_csv("Dept_Pred_HR_Attrition.csv")

# Create churn target variable
df['churn'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
drop_cols = [
    'Attrition',
    'Attrition Date',
    'EmployeeNumber',
    'EmployeeCount',
    'StandardHours',
    'Over18',
    'Random Number'
]
df = df.drop(columns=drop_cols)

# Store original categorical columns for the app
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=30,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Save the model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# Save feature names for the app
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save categorical columns info
with open('categorical_info.pkl', 'wb') as f:
    pickle.dump({
        'categorical_cols': categorical_cols,
        'df_sample': df.head(100)  # Save sample for getting unique values
    }, f)

print("Model training complete!")
print(f"Model accuracy on test set: {dt_model.score(X_test, y_test):.2%}")
