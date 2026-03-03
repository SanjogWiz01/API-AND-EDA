import pandas as pd

data = {
    "City":["Pokhara","Kathmandu","Butwal","Pokhara"]
}

df = pd.DataFrame(data)

encoded = pd.get_dummies(df["City"])

print(encoded)
# one hot = converting it into binary form
df_encoded = pd.get_dummies(df, columns=["City"])

print(df_encoded)
''' Pokhara = [0 0 1]
Kathmandu = [0 1 0]
Butwal = [1 0 0]
advantage = simple
disadvanatges = create many columns'''