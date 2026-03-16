import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

np.random.seed(42)

rows = 200

data = pd.DataFrame({
    "age": np.random.randint(18,60,rows),
    "salary": np.random.randint(20000,120000,rows),
    "experience": np.random.randint(0,20,rows),
    "city": np.random.choice(["kathmandu","pokhara","butwal","chitwan"],rows),
    "education": np.random.choice(["highschool","bachelor","master","phd"],rows),
    "purchased": np.random.randint(0,2,rows)
})

missing_age_index = np.random.choice(rows,15)
missing_salary_index = np.random.choice(rows,10)

data.loc[missing_age_index,"age"] = np.nan
data.loc[missing_salary_index,"salary"] = np.nan

age_imputer = SimpleImputer(strategy="mean")
salary_imputer = SimpleImputer(strategy="median")

data["age"] = age_imputer.fit_transform(data[["age"]])
data["salary"] = salary_imputer.fit_transform(data[["salary"]])

label_encoder = LabelEncoder()
data["education"] = label_encoder.fit_transform(data["education"])

one_hot = pd.get_dummies(data["city"],prefix="city")
data = pd.concat([data.drop("city",axis=1),one_hot],axis=1)

data["income_per_age"] = data["salary"] / data["age"]
data["experience_salary_ratio"] = data["experience"] / (data["salary"] + 1)

poly = PolynomialFeatures(degree=2,include_bias=False)
poly_features = poly.fit_transform(data[["age","experience"]])
poly_df = pd.DataFrame(poly_features,columns=["age","experience","age2","age_exp","exp2"])

data = pd.concat([data,poly_df[["age2","age_exp","exp2"]]],axis=1)

q1 = data["salary"].quantile(0.25)
q3 = data["salary"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

data["salary"] = np.where(data["salary"] < lower, lower, data["salary"])
data["salary"] = np.where(data["salary"] > upper, upper, data["salary"])

scaler = StandardScaler()
scaled_cols = ["age","salary","experience","income_per_age","experience_salary_ratio"]
data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

minmax = MinMaxScaler()
data[["age2","age_exp","exp2"]] = minmax.fit_transform(data[["age2","age_exp","exp2"]])

X = data.drop("purchased",axis=1)
y = data["purchased"]

selector = SelectKBest(score_func=f_regression,k=8)
X_selected = selector.fit_transform(X,y)

selected_columns = X.columns[selector.get_support()]

X_final = pd.DataFrame(X_selected,columns=selected_columns)

X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size=0.2,random_state=42)

print("Original Shape:",data.shape)
print("Selected Features:",selected_columns.tolist())
print("Train Shape:",X_train.shape)
print("Test Shape:",X_test.shape)
print(X_train.head())