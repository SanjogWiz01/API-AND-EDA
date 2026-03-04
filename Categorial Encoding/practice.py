# ===============================
# COMPLETE CATEGORICAL ENCODING DEMO

import pandas as pd
import numpy as np

# For sklearn encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher

# For advanced encoders
# Install if not available: pip install category_encoders
import category_encoders as ce

# -------------------------------
# 1. Create Sample Dataset
# -------------------------------

data = pd.DataFrame({
    'City': ['Kathmandu', 'Pokhara', 'Biratnagar', 'Kathmandu', 'Pokhara'],
    'Education': ['Bachelor', 'Master', 'PhD', 'High School', 'Bachelor'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Target': [1, 0, 1, 0, 1]   # Example target variable
})

print("Original Data:")
print(data)

# -------------------------------
# 2. Label Encoding
# -------------------------------

label_encoder = LabelEncoder()
data['City_Label'] = label_encoder.fit_transform(data['City'])

# -------------------------------
# 3. One-Hot Encoding
# -------------------------------

onehot = OneHotEncoder(sparse=False)
onehot_encoded = onehot.fit_transform(data[['City']])
onehot_df = pd.DataFrame(onehot_encoded, 
                         columns=onehot.get_feature_names_out(['City']))

# -------------------------------
# 4. Dummy Encoding
# -------------------------------

dummy_df = pd.get_dummies(data['City'], drop_first=True)

# -------------------------------
# 5. Ordinal Encoding (for ordered data)
# -------------------------------

education_order = ['High School', 'Bachelor', 'Master', 'PhD']
ordinal_encoder = OrdinalEncoder(categories=[education_order])
data['Education_Ordinal'] = ordinal_encoder.fit_transform(data[['Education']])

# -------------------------------
# 6. Binary Encoding
# -------------------------------

binary_encoder = ce.BinaryEncoder(cols=['City'])
binary_encoded = binary_encoder.fit_transform(data[['City']])

# -------------------------------
# 7. Target Encoding (Mean Encoding)
# -------------------------------

target_encoder = ce.TargetEncoder(cols=['City'])
target_encoded = target_encoder.fit_transform(data[['City']], data['Target'])

# -------------------------------
# 8. Leave-One-Out Encoding
# -------------------------------

loo_encoder = ce.LeaveOneOutEncoder(cols=['City'])
loo_encoded = loo_encoder.fit_transform(data[['City']], data['Target'])

# -------------------------------
# 9. Frequency Encoding
# -------------------------------

freq_encoding = data['City'].value_counts().to_dict()
data['City_Frequency'] = data['City'].map(freq_encoding)

# -------------------------------
# 10. Hash Encoding
# -------------------------------

hasher = FeatureHasher(n_features=4, input_type='string')
hashed_features = hasher.fit_transform(data['City'])
hashed_df = pd.DataFrame(hashed_features.toarray())

# -------------------------------
# 11. Weight of Evidence (WOE)
# -------------------------------

woe_encoder = ce.WOEEncoder(cols=['City'])
woe_encoded = woe_encoder.fit_transform(data[['City']], data['Target'])

# -------------------------------
# PRINT RESULTS
# -------------------------------

print("\nLabel Encoding:")
print(data[['City', 'City_Label']])

print("\nOne-Hot Encoding:")
print(onehot_df)

print("\nDummy Encoding:")
print(dummy_df)

print("\nOrdinal Encoding:")
print(data[['Education', 'Education_Ordinal']])

print("\nBinary Encoding:")
print(binary_encoded)

print("\nTarget Encoding:")
print(target_encoded)

print("\nLeave-One-Out Encoding:")
print(loo_encoded)

print("\nFrequency Encoding:")
print(data[['City', 'City_Frequency']])

print("\nHash Encoding:")
print(hashed_df)

print("\nWeight of Evidence Encoding:")
print(woe_encoded)