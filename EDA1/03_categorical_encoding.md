# 📙 File 3: Categorical Encoding

> **Series:** Feature Engineering for Data Science — 80/20 Guide  
> **Level:** Intermediate  
> **Prerequisite:** Files 1–2 (Data Types, EDA)

---

## 🧭 Why Encode Categorical Variables?

Machine learning models are **mathematical functions** — they require numbers as input. But real-world data is full of text categories like `"red"`, `"Manager"`, `"United States"`.

Categorical encoding is the process of **converting categories into numeric representations** in a way that:
1. Preserves the meaning of the category
2. Does not introduce false relationships
3. Works for the specific ML model you're using

**Choosing the wrong encoding can actively harm your model.** This file covers every major technique and when to use each.

---

## 🗺️ Encoding Decision Map

```
Is the variable ORDINAL (has natural order)?
│
├── YES → Ordinal Encoding
│
└── NO → Is it BINARY (only 2 categories)?
         │
         ├── YES → Binary Encoding (or Label Encoding)
         │
         └── NO → How many unique categories (cardinality)?
                  │
                  ├── LOW (< ~10) → One-Hot Encoding
                  │
                  ├── MEDIUM (10–50) → Target Encoding or Binary Encoding
                  │
                  └── HIGH (> 50) → Target Encoding, Frequency Encoding,
                                    or Hashing Encoding
```

---

## 1. Label Encoding

Assigns each category a unique integer.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
# red → 2, blue → 0, green → 1
```

**When to use:** Only for **tree-based models** (Decision Trees, Random Forest, XGBoost) and **binary** categories. Tree models split on thresholds and don't assume numeric order matters.

**When NOT to use:** Linear models, KNN, SVM — these models interpret `blue=0 < green=1 < red=2` as an ordering, which is false for nominal categories.

⚠️ **Pitfall:** Label encoding on nominal data with linear models introduces fake ordinal relationships.

---

## 2. One-Hot Encoding (OHE)

Creates one binary column per category.

```python
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue', 'red']})

# pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
#    color_green  color_red
# 0            0          1
# 1            0          0   ← 'blue' is baseline (dropped)
# 2            1          0
# 3            0          0
# 4            0          1

# scikit-learn version
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded = ohe.fit_transform(df[['color']])
```

**When to use:**
- Linear models (Logistic Regression, Linear Regression, SVM)
- Neural networks
- Low to medium cardinality (< ~15 unique categories)

**When NOT to use:**
- High cardinality (e.g., 500 cities) — creates too many columns ("curse of dimensionality")
- Very large datasets where memory matters

**The `drop_first=True` rule:** Always drop one dummy column to avoid **multicollinearity** (the "dummy variable trap"). The dropped column is the baseline that all others are compared against.

```python
# Without drop_first — WRONG (multicollinear):
# color_red + color_blue + color_green always = 1
# Any model can derive one column from the other two

# With drop_first — CORRECT:
# color_blue and color_green alone capture all information
```

---

## 3. Ordinal Encoding

Maps ordered categories to ordered integers **preserving rank**.

```python
from sklearn.preprocessing import OrdinalEncoder

education_order = [['High School', 'Bachelor', 'Master', 'PhD']]

oe = OrdinalEncoder(categories=education_order)
df['edu_encoded'] = oe.fit_transform(df[['education']])
# High School → 0, Bachelor → 1, Master → 2, PhD → 3

# Or manually with a mapping dict (clearer)
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['edu_encoded'] = df['education'].map(edu_map)
```

**When to use:** Any ordinal variable with a meaningful order. Works with all model types.

**Critical rule:** You must **define the order yourself** — don't let the encoder guess it alphabetically.

---

## 4. Target Encoding (Mean Encoding)

Replaces each category with the **mean of the target variable** for that category.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Chicago', 'NYC'],
    'salary': [80000, 70000, 90000, 60000, 75000, 62000, 85000]
})

# Calculate mean salary per city
target_mean = df.groupby('city')['salary'].mean()
print(target_mean)
# city
# Chicago    61000
# LA         72500
# NYC        85000

df['city_target_encoded'] = df['city'].map(target_mean)
```

**Why this is powerful:** It directly captures the relationship between the category and the target. High cardinality columns (like city with 500 values) are reduced to a single informative numeric column.

**The big problem — data leakage:**
Target encoding naively uses the same rows for both computing the mean and training the model. The model "sees the answer" during training → overfitting.

### Proper Target Encoding with Cross-Validation

```python
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

te = TargetEncoder(smoothing=10)  # smoothing prevents overfitting on rare categories
X_encoded = te.fit_transform(df[['city']], df['salary'])
```

The `smoothing` parameter blends the category mean with the global mean:
```
encoded_value = (count × category_mean + smoothing × global_mean) / (count + smoothing)
```
This prevents rare categories (1–2 observations) from getting extreme values.

**When to use:**
- High cardinality categoricals (cities, zip codes, product IDs)
- Tree-based models especially (they work very well with target encoding)
- **Always use inside a cross-validation loop to prevent leakage**

---

## 5. Frequency / Count Encoding

Replaces each category with how often it appears in the dataset.

```python
freq_map = df['city'].value_counts()          # count
pct_map  = df['city'].value_counts(normalize=True)  # proportion

df['city_freq_encoded'] = df['city'].map(freq_map)
df['city_pct_encoded']  = df['city'].map(pct_map)
```

**When to use:**
- When you believe **popularity of a category** is informative
- Quick and dirty baseline for high cardinality columns
- No risk of target leakage (doesn't use the target variable)

**Limitation:** Two different cities with the same frequency get the same encoding — the model can't distinguish them by content.

---

## 6. Binary Encoding

Converts categories to integers, then to binary, then splits each binary digit into a column.

```python
from category_encoders import BinaryEncoder

be = BinaryEncoder()
df_encoded = be.fit_transform(df[['city']])
```

For 500 cities: OHE needs 499 columns; Binary Encoding needs only **9 columns** (log₂(500) ≈ 9).

**When to use:**
- Medium–high cardinality (50–1000 categories)
- When OHE would create too many columns
- Tree-based models

---

## 7. Hashing Encoding

Maps categories to a fixed number of columns using a hash function. No need to know all categories in advance.

```python
from category_encoders import HashingEncoder

he = HashingEncoder(n_components=8)  # fixed 8 output columns regardless of cardinality
df_encoded = he.fit_transform(df[['city']])
```

**When to use:**
- Extremely high cardinality (100,000+ categories)
- Online learning (new categories arrive at inference time)
- When speed matters more than perfect accuracy

**Limitation:** Hash collisions — two different categories may hash to the same value.

---

## 8. Rare Category Handling

Before encoding, **always group rare categories** to prevent noise.

```python
# Find categories that appear less than 1% of the time
threshold = 0.01 * len(df)
rare_cats = df['job_title'].value_counts()[
    df['job_title'].value_counts() < threshold
].index

df['job_title_cleaned'] = df['job_title'].apply(
    lambda x: 'Other' if x in rare_cats else x
)

print(df['job_title_cleaned'].value_counts())
```

Why this matters:
- Rare categories add noise but little signal
- They hurt target encoding (mean based on 1–2 observations is unreliable)
- They bloat OHE columns

---

## Encoding Summary Table

| Encoding | Best for | Handles High Cardinality? | Risk of Leakage? |
|----------|----------|--------------------------|-----------------|
| Label Encoding | Ordinal or tree-based binary | N/A | No |
| One-Hot Encoding | Low cardinality nominal | No | No |
| Ordinal Encoding | Ordinal with known order | Yes | No |
| Target Encoding | High cardinality, trees | Yes | **Yes** — use CV |
| Frequency Encoding | Any cardinality | Yes | No |
| Binary Encoding | Medium–high cardinality | Yes | No |
| Hashing | Extreme cardinality | Yes | No |

---

## 🛠️ Practical Example: Encoding Pipeline

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Define column groups
nominal_cols = ['color', 'department']         # low cardinality
ordinal_cols = ['education']                    # ordered
high_card_cols = ['city']                       # high cardinality → use freq

# Ordinal category order
edu_categories = [['High School', 'Bachelor', 'Master', 'PhD']]

preprocessor = ColumnTransformer(transformers=[
    ('ohe',     OneHotEncoder(drop='first', sparse_output=False), nominal_cols),
    ('ordinal', OrdinalEncoder(categories=edu_categories), ordinal_cols),
])

# Frequency encode city BEFORE pipeline (or use TargetEncoder inside CV)
freq_map = df['city'].value_counts(normalize=True)
df['city_encoded'] = df['city'].map(freq_map)

# Full pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])
```

---

## 📋 Encoding Checklist

- [ ] Identify ordinal vs. nominal vs. binary columns
- [ ] Group rare categories into "Other" (< 1% threshold)
- [ ] Apply OHE for nominal low-cardinality columns (drop_first=True)
- [ ] Apply ordinal encoding with explicit order for ordinal columns
- [ ] Use frequency or target encoding for high-cardinality columns
- [ ] If using target encoding, implement inside cross-validation
- [ ] Check resulting feature matrix shape (should not explode from OHE)

---

## 📚 What's Next?

In **File 4**, we cover **Normalization and Scaling** — why raw numeric values need to be transformed for many ML algorithms and the right technique for every situation.

---

*Feature Engineering Series | File 3 of 5*
