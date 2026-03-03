from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample dataset
data = pd.DataFrame({
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Red', 'Blue']
})

# Create LabelEncoder instance
encoder = LabelEncoder()

# Fit and transform the 'Color' column
data['Color_encoded'] = encoder.fit_transform(data['Color'])

print(data)

# To see the mapping
label_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
print("Label Mapping:", label_mapping)
