# One Hot Encoding Example

import pandas as pd

# Sample dataset
data = {
    "City": ["Pokhara", "Kathmandu", "Butwal", "Pokhara"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Apply One Hot Encoding
encoded_df = pd.get_dummies(df, columns=["City"])

print("\nOne Hot Encoded Data:")
print(encoded_df)