import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'Age': [22, 25, np.nan, 30],
    'Salary': [50000, np.nan, 60000, 80000],
    'City': ['Kathmandu', 'Pokhara', np.nan, 'Lalitpur']
})

# 1. Detect missing values
print(df.isnull().sum())

# 2. Handle numerical missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 3. Handle categorical missing values
df['City'].fillna(df['City'].mode()[0], inplace=True)

# 4. Drop if necessary
df.dropna(inplace=True)

print(df)