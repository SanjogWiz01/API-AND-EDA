# 📗 File 2: Exploratory Data Analysis (EDA)

> **Series:** Feature Engineering for Data Science — 80/20 Guide  
> **Level:** Beginner → Intermediate  
> **Prerequisite:** File 1 (Data Types)

---

## 🧭 What is EDA?

Exploratory Data Analysis (EDA) is the **critical first step** of any data science project. Before you transform, encode, or model anything, you must *understand* your data.

EDA helps you answer:

- What does my data look like? (shape, types, samples)
- Are there missing values? Where? How many?
- What are the distributions of my features?
- Are there outliers?
- How are features related to each other?
- How are features related to the target variable?

> "No amount of sophisticated modelling can compensate for bad data or a poor understanding of it."

---

## 🗺️ The EDA Roadmap

```
1. Univariate Analysis    → Understand each feature in isolation
2. Bivariate Analysis     → Understand pairs of features (feature vs. target)
3. Multivariate Analysis  → Understand interactions among multiple features
4. Missing Value Analysis → Understand where and why data is missing
5. Outlier Detection      → Identify extreme values
```

---

## 1. Univariate Analysis — One Feature at a Time

### For Numeric Features

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('your_data.csv')

# Summary statistics
print(df['age'].describe())
# count    1000.000
# mean       35.200
# std        12.340
# min        18.000
# 25%        25.000
# 50%        33.000
# 75%        44.000
# max        85.000

# Histogram — shows distribution shape
df['age'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot — shows spread and outliers
df.boxplot(column='age')
plt.title('Box Plot of Age')
plt.show()
```

**What to look for:**
- Is the distribution **normal**, **skewed left**, **skewed right**, or **bimodal**?
- Are there suspicious peaks at 0 or round numbers (data quality issue)?
- How wide is the spread (std vs. mean)?

### For Categorical Features

```python
# Value counts — frequency of each category
print(df['job_title'].value_counts())
print(df['job_title'].value_counts(normalize=True))  # proportions

# Bar chart
df['job_title'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Job Title Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**What to look for:**
- How many unique categories are there (cardinality)?
- Are there dominant categories? (80% in one bucket = low signal)
- Are there rare categories that may need grouping?
- Are there obvious typos or duplicates? (e.g., "New York" vs "new york" vs "NY")

---

## 2. Bivariate Analysis — Feature vs. Target

This is where EDA becomes most powerful for feature engineering. You are looking for **features that have a strong relationship with your target**.

### Numeric Feature vs. Numeric Target (Regression)

```python
# Scatter plot
df.plot.scatter(x='experience_years', y='salary')
plt.title('Experience vs Salary')
plt.show()

# Correlation coefficient
print(df['experience_years'].corr(df['salary']))
# 0.73 — strong positive correlation

# Correlation heatmap for all numeric columns
corr_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

### Categorical Feature vs. Numeric Target

```python
# Box plot per category — great for seeing group differences
df.boxplot(column='salary', by='job_title')
plt.title('Salary by Job Title')
plt.suptitle('')  # remove default pandas title
plt.xticks(rotation=45)
plt.show()

# Or using seaborn for cleaner output
sns.boxplot(data=df, x='job_title', y='salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Categorical Feature vs. Categorical Target (Classification)

```python
# Crosstab
ct = pd.crosstab(df['job_title'], df['churned'], normalize='index')
print(ct)

# Stacked bar
ct.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Job Title')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 3. Multivariate Analysis — Multiple Features Together

```python
# Pair plot — scatter matrix of all numeric features
sns.pairplot(df[['age', 'salary', 'experience_years', 'churned']], hue='churned')
plt.suptitle('Pair Plot', y=1.02)
plt.show()
```

**Important:** Pair plots can be slow on large datasets. Sample first:
```python
df_sample = df.sample(500, random_state=42)
sns.pairplot(df_sample[['age', 'salary', 'experience_years', 'churned']], hue='churned')
```

---

## 4. Missing Value Analysis

This is non-negotiable. Missing values are everywhere, and understanding them guides how you handle them.

### Types of Missingness

| Type | What it means | Example |
|------|--------------|---------|
| **MCAR** — Missing Completely at Random | Missingness is unrelated to any data | Random sensor failure |
| **MAR** — Missing at Random | Missingness depends on observed data | Income missing more for younger people |
| **MNAR** — Missing Not at Random | Missingness depends on the missing value itself | High earners don't report income |

Understanding the type matters because:
- **MCAR/MAR**: Safe to impute (fill in) with statistical methods
- **MNAR**: Imputation can introduce bias; the missingness itself is often a feature

### Detecting Missing Values

```python
# Total missing per column
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

# Percentage missing
missing_pct = df.isnull().mean() * 100
print(missing_pct.sort_values(ascending=False))

# Visual heatmap of missing values
import missingno as msno   # pip install missingno
msno.matrix(df)
plt.show()

# Is missingness correlated between columns?
msno.heatmap(df)
plt.show()
```

### Missing Value Rules of Thumb

| % Missing | Action |
|-----------|--------|
| < 5%      | Safe to drop rows or simple impute (mean/median) |
| 5–30%     | Impute carefully; consider adding `feature_is_missing` indicator |
| > 30%     | Consider dropping the column or using advanced imputation |
| > 70%     | Usually drop the column |

```python
# Add "was missing" binary indicator BEFORE imputing
df['income_was_missing'] = df['income'].isnull().astype(int)

# Then impute
df['income'] = df['income'].fillna(df['income'].median())
```

---

## 5. Outlier Detection

Outliers are data points that are **far from the rest of the distribution**. They can be:
- **Genuine** (a real billionaire in a salary dataset)
- **Data errors** (age of 999, weight of -5)
- **Important signals** (fraud transaction)

### Method 1: IQR (Interquartile Range)

```python
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
print(f"Number of outliers: {len(outliers)}")
print(f"Outlier range: below {lower:.2f} or above {upper:.2f}")
```

### Method 2: Z-Score

```python
from scipy import stats

z_scores = stats.zscore(df['salary'])
outliers_z = df[abs(z_scores) > 3]
print(f"Z-score outliers: {len(outliers_z)}")
```

### Visualizing Outliers

```python
# Box plot
df.boxplot(column='salary')
plt.title('Salary Box Plot — Outliers Shown')
plt.show()

# Scatter with highlighted outliers
df['is_outlier'] = abs(stats.zscore(df['salary'])) > 3
colors = df['is_outlier'].map({True: 'red', False: 'steelblue'})
df['salary'].reset_index(drop=True).plot(
    kind='scatter', x=df.index, y='salary', c=colors
)
```

### What to Do With Outliers

| Situation | Action |
|-----------|--------|
| Data error (impossible value) | Remove or correct |
| Genuine extreme value, sensitive model | Cap (winsorize) or log-transform |
| Genuine extreme value, robust model | Keep as-is |
| Outlier is the signal (fraud detection) | Keep! It's what you're predicting |

```python
# Winsorization — cap at 5th and 95th percentile
lower_cap = df['salary'].quantile(0.05)
upper_cap = df['salary'].quantile(0.95)
df['salary_capped'] = df['salary'].clip(lower=lower_cap, upper=upper_cap)
```

---

## 📋 EDA Checklist (Use This Every Project)

**Shape & Types**
- [ ] `df.shape`, `df.dtypes`, `df.info()`
- [ ] Identify numeric, categorical, datetime, boolean, text columns

**Univariate**
- [ ] Histograms for all numeric columns
- [ ] Value counts for all categorical columns
- [ ] Identify skewed distributions

**Bivariate**
- [ ] Correlation matrix for numeric features
- [ ] Box plots: categorical features vs. numeric target
- [ ] Scatter plots: numeric features vs. numeric target

**Missing Values**
- [ ] `df.isnull().sum()` — count missing per column
- [ ] Decide: drop, impute, or create indicator
- [ ] Document missing value decisions

**Outliers**
- [ ] Box plots for numeric features
- [ ] IQR or Z-score analysis
- [ ] Decide: remove, cap, or transform

---

## 🔑 Key EDA Takeaways for Feature Engineering

- **Skewed numeric features** → consider log transform (see File 4)
- **Categorical with many rare categories** → group them into "Other" before encoding (see File 3)
- **Missing values** → create binary "was_missing" indicator BEFORE imputing
- **High correlation (> 0.9) between two features** → one may be redundant; consider dropping
- **Low variance feature** → may carry little signal; consider dropping
- **Date columns** → extract day, month, year, day of week, is_weekend (see File 1)

---

## 📚 What's Next?

In **File 3**, we cover **Categorical Encoding** — how to transform text categories into numbers that machine learning models can understand, without accidentally introducing false relationships.

---

*Feature Engineering Series | File 2 of 5*
