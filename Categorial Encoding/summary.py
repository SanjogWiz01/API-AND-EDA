import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# -----------------------------
# 1 Create Sample Election Dataset
# -----------------------------

data = {
    "Party": ["Congress","UML","Maoist","Congress","UML","Maoist"],
    "Region": ["Urban","Rural","Urban","Rural","Urban","Rural"],
    "Education": ["HighSchool","Bachelor","Master","Bachelor","HighSchool","Master"],
    "Voted": [1,0,1,1,0,1]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)


# -----------------------------
# 2 Label Encoding
# -----------------------------

label_encoder = LabelEncoder()
df["Party_Label"] = label_encoder.fit_transform(df["Party"])

print("\nLabel Encoding (Party)")
print(df[["Party","Party_Label"]])


# -----------------------------
# 3 One Hot Encoding
# -----------------------------

onehot_df = pd.get_dummies(df, columns=["Region"])

print("\nOne Hot Encoding (Region)")
print(onehot_df)


# -----------------------------
# 4 Ordinal Encoding
# -----------------------------

ordinal_encoder = OrdinalEncoder(
    categories=[["HighSchool","Bachelor","Master"]]
)

df["Education_Ordinal"] = ordinal_encoder.fit_transform(df[["Education"]])

print("\nOrdinal Encoding (Education)")
print(df[["Education","Education_Ordinal"]])


# -----------------------------
# 5 Frequency Encoding
# -----------------------------

freq_map = df["Party"].value_counts()

df["Party_Frequency"] = df["Party"].map(freq_map)

print("\nFrequency Encoding (Party)")
print(df[["Party","Party_Frequency"]])


# -----------------------------
# 6 Target Encoding
# -----------------------------

target_map = df.groupby("Party")["Voted"].mean()

df["Party_Target"] = df["Party"].map(target_map)

print("\nTarget Encoding (Party)")
print(df[["Party","Party_Target"]])