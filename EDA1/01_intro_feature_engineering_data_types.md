# 📘 File 1: Introduction to Feature Engineering & Data Types

> **Series:** Feature Engineering for Data Science — 80/20 Guide  
> **Level:** Beginner → Intermediate  
> **Prerequisite:** Basic Python, Pandas basics

---

## 🧭 What is Feature Engineering?

Feature engineering is the process of **transforming raw data into features** that better represent the underlying patterns to your machine learning model — leading to improved model accuracy on unseen data.

> "Applied machine learning is basically feature engineering." — Andrew Ng

Think of it like this: your raw data is crude oil. Feature engineering is the refining process. The quality of your features often matters **more than the choice of model**.

---

## 🎯 Why Does Feature Engineering Matter? (The 80/20 Rule)

In practice, **80% of the value in a machine learning project** comes from good data preparation and feature engineering, while only about 20% comes from model selection and tuning.

| What you spend time on | Impact on final result |
|------------------------|------------------------|
| Data cleaning          | Very High              |
| Feature engineering    | **Very High**          |
| Model selection        | Medium                 |
| Hyperparameter tuning  | Low–Medium             |

The 80/20 principle tells us: **master feature engineering first**, and you will outperform most practitioners who jump straight to model tuning.

---

## 📦 The Feature Engineering Pipeline (Bird's Eye View)

```
Raw Data
   │
   ▼
[ 1. Understand your data types ]
   │
   ▼
[ 2. Handle missing values ]
   │
   ▼
[ 3. Encode categorical variables ]
   │
   ▼
[ 4. Scale / Normalize numeric features ]
   │
   ▼
[ 5. Create new features (feature creation) ]
   │
   ▼
[ 6. Select important features (feature selection) ]
   │
   ▼
Clean, ML-ready Feature Matrix (X)
```

This guide covers **Step 1** in depth. The other steps are covered in Files 2–5.

---

## 🗂️ Data Types — The Foundation of Everything

Before you can engineer features, you must understand what *kind* of data you're working with. Every transformation you apply depends on data type.

### 1. Numeric (Quantitative) Data

Data that is measurable and arithmetic operations make sense.

#### 1a. Continuous
- Can take **any value** in a range.
- Examples: height (1.73 m), temperature (36.6°C), salary ($54,321.50)
- Operations: addition, subtraction, mean, standard deviation — all valid.

#### 1b. Discrete
- Can only take **integer/countable values**.
- Examples: number of children (0, 1, 2, 3...), number of items in a cart.
- Operations: mean and median still valid; "2.5 children" is mathematically OK as a statistic.

```python
import pandas as pd

df = pd.DataFrame({
    'height_cm': [170.5, 182.0, 158.3],   # continuous
    'num_children': [0, 2, 1]              # discrete
})

# Check dtypes
print(df.dtypes)
# height_cm      float64
# num_children     int64
```

---

### 2. Categorical (Qualitative) Data

Data that represents **groups or labels**. Arithmetic doesn't apply directly.

#### 2a. Nominal
- Categories with **no inherent order**.
- Examples: color (red, blue, green), country, job title, blood type.
- "Red > Blue" is meaningless.

#### 2b. Ordinal
- Categories with a **meaningful order**, but differences between levels are not necessarily equal.
- Examples: education level (High School < Bachelor's < Master's < PhD), satisfaction rating (poor < fair < good < excellent).
- "PhD > Bachelor's" makes sense; the *gap* between levels is unclear.

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green'],           # nominal
    'education': ['Bachelor', 'PhD', 'Master']   # ordinal
})

# Pandas Categorical with order
from pandas.api.types import CategoricalDtype

edu_order = CategoricalDtype(
    categories=['High School', 'Bachelor', 'Master', 'PhD'],
    ordered=True
)
df['education'] = df['education'].astype(edu_order)

print(df['education'] > 'Bachelor')
# 0    False
# 1     True
# 2     True
```

---

### 3. Binary Data

A special case of categorical with exactly **two possible values**.

- Examples: is_fraud (0/1), has_purchased (True/False), gender (if binary encoded).
- Stored as bool or int in pandas.

```python
df['is_fraud'] = [True, False, False]
df['has_purchased'] = [1, 0, 1]
```

---

### 4. Date and Time Data

Temporal data — one of the **richest sources of features**.

- Examples: order date, login timestamp, date of birth.
- Raw datetime strings are not directly useful to models.
- You extract components or derive features.

```python
df['order_date'] = pd.to_datetime(['2024-01-15', '2024-03-22', '2024-07-01'])

# Extract useful components
df['day_of_week']  = df['order_date'].dt.dayofweek  # 0=Monday
df['month']        = df['order_date'].dt.month
df['quarter']      = df['order_date'].dt.quarter
df['is_weekend']   = df['order_date'].dt.dayofweek >= 5
df['days_since_epoch'] = (df['order_date'] - pd.Timestamp('1970-01-01')).dt.days
```

---

### 5. Text Data

Free-form strings that require NLP-based feature extraction.

- Examples: product reviews, email bodies, social media posts.
- Common features: word count, character count, sentiment score, TF-IDF vectors.

```python
df['review'] = ['Great product!', 'Terrible experience.', 'It was okay.']

# Basic text features
df['word_count'] = df['review'].str.split().str.len()
df['char_count'] = df['review'].str.len()
df['has_exclamation'] = df['review'].str.contains('!').astype(int)
```

---

### 6. Image & Audio Data (Advanced)

- Require specialized pipelines (CNNs for images, spectrograms for audio).
- Out of scope for this series, but be aware they exist.

---

## 🔍 How to Quickly Profile Your Data Types

```python
import pandas as pd

df = pd.read_csv('your_dataset.csv')

# Step 1: Check shapes and types
print(df.shape)         # (rows, columns)
print(df.dtypes)        # dtype per column
print(df.info())        # non-null counts + dtypes

# Step 2: Separate by type
numeric_cols     = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols        = df.select_dtypes(include=['bool']).columns.tolist()
datetime_cols    = df.select_dtypes(include=['datetime64']).columns.tolist()

print("Numeric:    ", numeric_cols)
print("Categorical:", categorical_cols)
print("Boolean:    ", bool_cols)
print("Datetime:   ", datetime_cols)

# Step 3: Basic stats per type
print(df[numeric_cols].describe())
print(df[categorical_cols].describe())

# Step 4: Check for hidden categoricals (numeric with few unique values)
for col in numeric_cols:
    n_unique = df[col].nunique()
    if n_unique < 10:
        print(f"⚠️  '{col}' has only {n_unique} unique values — might be categorical")
```

---

## 🚩 Common Mistakes When Ignoring Data Types

| Mistake | Problem |
|---------|---------|
| Treating ordinal as nominal | You lose the order information |
| Treating nominal as numeric | Model sees spurious math (e.g., red=1 > blue=0) |
| Not parsing datetimes | Loses year, month, seasonality signals |
| Ignoring text as a feature | Misses rich signal |
| Assuming int = truly numeric | Zip codes, IDs are integers but not numeric features |

**Golden rule:** Always ask — *"Does arithmetic on this column make sense?"* If not, treat it as categorical.

---

## 🛠️ Practical Checklist — Step 1 of Any Project

- [ ] Load data and call `df.info()` and `df.dtypes`
- [ ] Parse date columns with `pd.to_datetime()`
- [ ] Identify columns with low cardinality that may be categorical
- [ ] Check for ID columns (should be dropped or used only for joining)
- [ ] Identify target variable (y) and separate from features (X)
- [ ] Document each column's semantic type vs. its pandas dtype

---

## 📚 What's Next?

In **File 2**, we cover **Exploratory Data Analysis (EDA)** — the practice of visually and statistically understanding your data before any transformation. EDA informs every decision in Files 3–5.

---

*Feature Engineering Series | File 1 of 5*
