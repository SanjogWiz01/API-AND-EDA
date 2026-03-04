# ======================================
# COMPLETE DATA PREPROCESSING PIPELINE
# ======================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    OneHotEncoder, OrdinalEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------
# 1. CREATE SAMPLE DATASET
# --------------------------------------

data = pd.DataFrame({
    'Age': [22, 25, np.nan, 35, 40, 29],
    'Salary': [25000, 32000, 28000, np.nan, 50000, 42000],
    'City': ['Kathmandu', 'Pokhara', 'Kathmandu', 'Biratnagar', np.nan, 'Pokhara'],
    'Education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'High School', 'Master'],
    'Purchased': [0, 1, 0, 1, 1, 0]
})

print("Original Data:")
print(data)

# --------------------------------------
# 2. HANDLE MISSING VALUES
# --------------------------------------

numeric_features = ['Age', 'Salary']
categorical_features = ['City']
ordinal_features = ['Education']

# Ordinal order
education_order = ['High School', 'Bachelor', 'Master', 'PhD']

# --------------------------------------
# 3. NUMERIC PIPELINE
# --------------------------------------

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# --------------------------------------
# 4. CATEGORICAL PIPELINE (Nominal)
# --------------------------------------

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# --------------------------------------
# 5. ORDINAL PIPELINE
# --------------------------------------

ordinal_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[education_order]))
])

# --------------------------------------
# 6. COMBINE ALL PIPELINES
# --------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
    ('ord', ordinal_pipeline, ordinal_features)
])

# --------------------------------------
# 7. SPLIT DATA
# --------------------------------------

X = data.drop('Purchased', axis=1)
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------
# 8. COMPLETE PIPELINE WITH MODEL
# --------------------------------------

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier())
])

# --------------------------------------
# 9. TRAIN MODEL
# --------------------------------------

model_pipeline.fit(X_train, y_train)

# --------------------------------------
# 10. TRANSFORMED TRAIN DATA (VIEW)
# --------------------------------------

X_processed = preprocessor.fit_transform(X_train)

print("\nProcessed Training Data Shape:", X_processed.shape)
print("\nModel Training Complete!")