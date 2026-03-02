import pandas as pd
data = {
    "Age": [22, 25, 30, 35],
    "Salary": [20000, 30000, 40000, 50000],
    "City": ["Pokhara", "Kathmandu", "Pokhara", "Butwal"],
    "Bought": [0, 1, 1, 0]
}

df = pd.DataFrame(data)

# print
print(df.head())
print('/n Describe the comdition of data ',df.describe())
 # making features vs target
x=df.drop("City",axis=1)
y=df["Bought"]
print('FEATURES',x)
print("Target",y)
# identify the data types
print("features types",x.dtypes)
print('\n target',y.dtype)
