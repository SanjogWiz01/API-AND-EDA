import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

# ==============================
# 2. LOAD DATA
# ==============================

def load_data():
    # Example dataset (replace with CSV)
    df = pd.DataFrame({
        'Age': [20, 25, np.nan, 35, 40],
        'Salary': [20000, 50000, 60000, np.nan, 100000],
        'City': ['Kathmandu', 'Pokhara', 'Lalitpur', 'Kathmandu', np.nan],
        'Experience': [1, 3, 5, 7, 10]
    })
    return df

# ==============================
# 3. BASIC EDA
# ==============================

def basic_eda(df):
    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== DESCRIPTION =====")
    print(df.describe())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== UNIQUE VALUES =====")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

# ==============================
# 4. HANDLE MISSING VALUES
# ==============================

def handle_missing_values(df):
    # Numerical columns
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# ==============================
# 5. DATA TYPE HANDLING
# ==============================

def handle_data_types(df):
    # Convert categorical to category type
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    return df

# ==============================
# 6. CATEGORICAL ENCODING
# ==============================

def encode_categorical(df):
    df_encoded = df.copy()

    # Label Encoding (for simplicity)
    le = LabelEncoder()

    for col in df_encoded.select_dtypes(include='category').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # One-hot encoding (optional)
    # df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    return df_encoded

# ==============================
# 7. FEATURE CREATION
# ==============================

def create_features(df):
    df_new = df.copy()

    # Example feature engineering
    df_new['Age_per_Experience'] = df_new['Age'] / (df_new['Experience'] + 1)

    df_new['Salary_per_Age'] = df_new['Salary'] / (df_new['Age'] + 1)

    return df_new

# ==============================
# 8. FEATURE SCALING
# ==============================

def scale_features(df):
    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled_array, columns=df.columns)

    return df_scaled

# ==============================
# 9. FEATURE SELECTION
# ==============================

def select_features(df):
    # Assume Salary is target
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    selector = SelectKBest(score_func=f_regression, k='all')
    X_new = selector.fit_transform(X, y)

    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })

    print("\n===== FEATURE IMPORTANCE =====")
    print(scores.sort_values(by='Score', ascending=False))

    return X

# ==============================
# 10. PIPELINE EXECUTION
# ==============================

def main():
    print("\n🚀 Starting Feature Engineering Pipeline...\n")

    # Load data
    df = load_data()

    # EDA
    basic_eda(df)

    # Missing values
    df = handle_missing_values(df)

    # Data types
    df = handle_data_types(df)

    # Encoding
    df = encode_categorical(df)

    # Feature creation
    df = create_features(df)

    print("\n===== DATA AFTER FEATURE ENGINEERING =====")
    print(df.head())

    # Feature selection
    select_features(df)

    # Scaling
    df_scaled = scale_features(df)

    print("\n===== FINAL SCALED DATA =====")
    print(df_scaled.head())

    print("\n✅ Pipeline Completed Successfully!")

# ==============================
# RUN SCRIPT
# ==============================

if __name__ == "__main__":
    main()