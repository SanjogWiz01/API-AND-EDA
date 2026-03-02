import pandas as pd

data = {
    "Age":[18,25,35,60]
}

df = pd.DataFrame(data)

df["Age_Group"] = ["Young","Adult","Adult","Senior"]

print(df)
