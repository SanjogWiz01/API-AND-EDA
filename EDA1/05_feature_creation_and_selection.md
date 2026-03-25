# 📓 File 5: Feature Creation & Feature Selection

> **Series:** Feature Engineering for Data Science — 80/20 Guide  
> **Level:** Intermediate → Advanced  
> **Prerequisite:** Files 1–4

---

## 🧭 Overview

This final file covers the two most creative and high-leverage parts of feature engineering:

- **Feature Creation**: Crafting new columns from existing data that give the model new signal
- **Feature Selection**: Identifying and removing features that are redundant, noisy, or harmful

Together, they represent the "craft" of feature engineering — where domain knowledge meets data science.

---

# PART A: Feature Creation

## Why Create New Features?

Raw features often don't directly capture what matters. Consider:
- Raw: `date_of_birth` → Hard to use directly
- Engineered: `age = today - date_of_birth` → Much clearer signal

Or in e-commerce:
- Raw: `price` and `cost`
- Engineered: `profit_margin = (price - cost) / price` → Captures what really drives decisions

Good feature engineering often doubles or triples model performance without changing the algorithm at all.

---

## 1. Mathematical Combinations

Create features by combining existing numeric columns.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'revenue':     [10000, 25000, 8000, 50000],
    'cost':        [7000,  18000, 9000, 30000],
    'num_orders':  [50, 100, 40, 200],
    'num_returns': [5, 8, 10, 15]
})

# Ratios
df['profit_margin']  = (df['revenue'] - df['cost']) / df['revenue']
df['return_rate']    = df['num_returns'] / df['num_orders']
df['avg_order_value'] = df['revenue'] / df['num_orders']

# Differences
df['gross_profit'] = df['revenue'] - df['cost']

# Products
df['revenue_x_orders'] = df['revenue'] * df['num_orders']  # volume signal

print(df[['profit_margin', 'return_rate', 'avg_order_value']].round(3))
```

**Rule of thumb:** Think about what a domain expert would compute by hand. Those business metrics are usually good features.

---

## 2. Date and Time Features

Datetime columns are goldmines of features.

```python
df['order_date'] = pd.to_datetime(df['order_date'])

# Calendar features
df['year']        = df['order_date'].dt.year
df['month']       = df['order_date'].dt.month
df['day']         = df['order_date'].dt.day
df['day_of_week'] = df['order_date'].dt.dayofweek   # 0=Monday
df['quarter']     = df['order_date'].dt.quarter
df['week_of_year']= df['order_date'].dt.isocalendar().week.astype(int)

# Boolean flags
df['is_weekend']  = df['order_date'].dt.dayofweek >= 5
df['is_month_end']= df['order_date'].dt.is_month_end
df['is_quarter_end'] = df['order_date'].dt.is_quarter_end

# Time since reference point
reference_date = pd.Timestamp('2020-01-01')
df['days_since_launch'] = (df['order_date'] - reference_date).dt.days

# Time until an event
deadline = pd.Timestamp('2025-12-31')
df['days_until_deadline'] = (deadline - df['order_date']).dt.days
```

### Cyclical Encoding for Periodic Features

Month, day of week, and hour are **cyclical** — December (12) and January (1) are close, but numerically they look far apart. Fix this with sine/cosine encoding:

```python
# Month: 1–12 cyclical
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of week: 0–6 cyclical
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Hour: 0–23 cyclical
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

Now January (month=1) and December (month=12) will be numerically close in the (sin, cos) space.

---

## 3. Text Features (Basic NLP)

```python
df['review_text'] = [
    'This product is absolutely amazing!!! Love it.',
    'Terrible. Broke on first use. Very disappointed.',
    'It is okay I guess.',
]

# Length features
df['char_count']     = df['review_text'].str.len()
df['word_count']     = df['review_text'].str.split().str.len()
df['avg_word_len']   = df['review_text'].str.split().apply(
    lambda words: np.mean([len(w) for w in words]) if words else 0
)

# Punctuation features
df['exclamation_count'] = df['review_text'].str.count('!')
df['question_count']    = df['review_text'].str.count(r'\?')
df['caps_ratio']        = df['review_text'].apply(
    lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
)

# Keyword flags
positive_words = ['amazing', 'love', 'excellent', 'great', 'best']
negative_words = ['terrible', 'broken', 'disappointed', 'awful', 'worst']

df['positive_word_count'] = df['review_text'].str.lower().apply(
    lambda x: sum(word in x for word in positive_words)
)
df['negative_word_count'] = df['review_text'].str.lower().apply(
    lambda x: sum(word in x for word in negative_words)
)
```

### TF-IDF for Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['review_text'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)
```

---

## 4. Interaction Features

Capture combined effects of two variables that individually don't tell the full story.

```python
# Polynomial features (all combinations up to degree 2)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df[['age', 'experience_years', 'salary']])

feature_names = poly.get_feature_names_out(['age', 'experience_years', 'salary'])
print(feature_names)
# ['age', 'experience_years', 'salary',
#  'age experience_years', 'age salary', 'experience_years salary']

# Manual interaction (often better — use domain knowledge)
df['age_x_experience'] = df['age'] * df['experience_years']
df['is_senior_high_earner'] = ((df['experience_years'] > 10) & (df['salary'] > 100000)).astype(int)
```

---

## 5. Binning (Discretization)

Convert continuous variables into categorical bins.

```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['Under 18', '18-35', '35-50', '50-65', '65+']
)

# Equal-frequency bins (quantile-based)
df['salary_quartile'] = pd.qcut(df['salary'], q=4,
    labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
)

# Manual thresholds based on domain knowledge
df['risk_tier'] = pd.cut(df['credit_score'],
    bins=[0, 580, 670, 740, 800, 850],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
)
```

**When to use binning:**
- When you believe the relationship is non-linear (e.g., "middle-aged" people behave differently from both young and old)
- When you want to reduce the impact of outliers
- When the exact value matters less than the tier (credit scoring)

**Caveat:** Binning discards information within bins. Only use when you have a good reason.

---

## 6. Aggregation Features (Group Statistics)

Powerful for tabular data with a grouping structure.

```python
# For each customer, compute statistics across their transactions
df_transactions = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2, 3],
    'amount': [50, 120, 30, 200, 80, 500],
    'category': ['food', 'electronics', 'food', 'clothing', 'food', 'electronics']
})

# Group-level features
customer_stats = df_transactions.groupby('customer_id')['amount'].agg(
    total_spend='sum',
    avg_spend='mean',
    max_spend='max',
    min_spend='min',
    spend_std='std',
    num_transactions='count'
).reset_index()

# Category diversity
category_diversity = df_transactions.groupby('customer_id')['category'].nunique()
customer_stats['category_diversity'] = customer_stats['customer_id'].map(category_diversity)

print(customer_stats)
```

---

# PART B: Feature Selection

## Why Select Features?

More features ≠ better model. Too many features leads to:
- **Overfitting**: Model learns noise instead of signal
- **Slow training**: More computation
- **Multicollinearity**: Redundant features confuse linear models
- **Curse of dimensionality**: Performance degrades in high dimensions

Feature selection removes features that add noise without adding signal.

---

## Method 1: Variance Threshold (Remove Constant or Near-Constant Features)

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Remove features with < 1% variance
X_selected = selector.fit_transform(X)

# Which features were removed?
removed = X.columns[~selector.get_support()].tolist()
print("Removed features:", removed)
```

A column that is always 0 (or always the same value) contains **zero information**. Remove it.

---

## Method 2: Correlation Analysis (Remove Redundant Features)

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr().abs()

# Plot heatmap
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Find pairs with correlation > 0.9 (highly redundant)
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
print("Highly correlated (drop one of each pair):", to_drop)

df_reduced = df.drop(columns=to_drop)
```

If two features are 95% correlated, they carry almost the same information. Keep one, drop the other.

---

## Method 3: Univariate Statistical Tests

Test each feature's relationship with the target independently.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2

# For classification — F-test (ANOVA)
selector_f = SelectKBest(score_func=f_classif, k=10)
X_selected_f = selector_f.fit_transform(X_numeric, y)

# For classification — Mutual Information (captures non-linear relationships)
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected_mi = selector_mi.fit_transform(X_numeric, y)

# Show feature scores
scores = pd.DataFrame({
    'Feature': X.columns,
    'F-score': selector_f.scores_,
    'MI-score': selector_mi.scores_
}).sort_values('MI-score', ascending=False)

print(scores.head(15))
```

**When to use:**
- Quick first pass to remove obviously useless features
- Very fast, scales to large datasets

**Limitation:** Evaluates each feature in isolation — doesn't account for feature interactions.

---

## Method 4: Feature Importance from Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get importances
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot
importance_df.head(20).plot(
    kind='barh', x='Feature', y='Importance',
    color='steelblue', figsize=(10, 8)
)
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Select top N features
top_features = importance_df.head(15)['Feature'].tolist()
X_train_selected = X_train[top_features]
X_test_selected  = X_test[top_features]
```

---

## Method 5: Recursive Feature Elimination (RFE)

Iteratively removes the least important feature and rebuilds the model.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

# Fixed number of features
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_].tolist()
print("RFE selected features:", selected_features)

# Cross-validated RFE (automatically finds optimal number of features)
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,
    scoring='accuracy'
)
rfecv.fit(X_train, y_train)
print(f"Optimal number of features: {rfecv.n_features_}")
```

---

## Method 6: L1 Regularization (Lasso) — Embedded Method

Lasso regression adds an L1 penalty that **drives some coefficients to exactly zero**, performing feature selection automatically.

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# Find optimal alpha via cross-validation
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

print(f"Best alpha: {lasso.alpha_:.4f}")

# Features with non-zero coefficients
selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X_train_scaled)

selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask].tolist()
print(f"Selected {len(selected_features)} features:", selected_features)

# Visualize coefficients
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lasso.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df)
```

---

## Feature Selection Summary

| Method | Speed | Model Required | Captures Interactions | Best For |
|--------|-------|---------------|----------------------|----------|
| Variance Threshold | Fastest | No | No | Remove constants |
| Correlation | Fast | No | No | Remove redundant pairs |
| Univariate (F-test, MI) | Fast | No | Partial | Quick screening |
| Tree Importance | Medium | Yes (tree) | Yes | General use |
| RFE | Slow | Yes (any) | Yes | Accurate selection |
| L1 (Lasso) | Medium | Yes (linear) | No | Linear model selection |

---

## 🏁 Complete Feature Engineering Workflow

```python
# 1. Load and inspect
df = pd.read_csv('data.csv')
print(df.info())

# 2. Parse dates, fix types
df['date'] = pd.to_datetime(df['date'])

# 3. EDA — understand distributions, missing, outliers
# (See File 2)

# 4. Handle missing values
df['income_missing'] = df['income'].isnull().astype(int)
df['income'] = df['income'].fillna(df['income'].median())

# 5. Encode categoricals
# (See File 3)

# 6. Create new features
df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
df['profit_margin'] = (df['revenue'] - df['cost']) / df['revenue']
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

# 7. Split data
from sklearn.model_selection import train_test_split
X = df.drop(columns=['target', 'id', 'date'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Scale (inside pipeline)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")

# 9. Feature selection — inspect and prune
# Use RandomForest importances from pipeline
importances = pipeline.named_steps['model'].feature_importances_
imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
print(imp_df.sort_values('Importance', ascending=False).head(15))
```

---

## 📋 Final Checklist — Feature Engineering End-to-End

**Foundation (File 1)**
- [ ] Understand all column data types (numeric, ordinal, nominal, datetime, text)
- [ ] Parse datetimes, fix wrong dtypes

**EDA (File 2)**
- [ ] Histograms / value counts for all columns
- [ ] Correlation matrix
- [ ] Missing value analysis
- [ ] Outlier detection

**Encoding (File 3)**
- [ ] Group rare categories
- [ ] OHE for nominal; Ordinal for ordered; Target/Frequency for high-cardinality

**Scaling (File 4)**
- [ ] Log-transform skewed numeric features
- [ ] Choose scaler based on model type
- [ ] Fit scaler on train only (use Pipeline)

**Feature Creation (File 5A)**
- [ ] Compute ratios, differences, products from domain knowledge
- [ ] Extract datetime components; cyclical encode month/hour/day
- [ ] Create text length and sentiment features
- [ ] Add interaction features where domain makes sense

**Feature Selection (File 5B)**
- [ ] Remove zero-variance features
- [ ] Remove highly correlated features (> 0.9)
- [ ] Use model importances or RFE to select top features
- [ ] Validate selection with cross-validation

---

## 🎓 The 80/20 of Feature Engineering — Summary

| Priority | Technique | Impact |
|----------|-----------|--------|
| ⭐⭐⭐⭐⭐ | Handle missing values properly | Critical |
| ⭐⭐⭐⭐⭐ | Correct data types | Critical |
| ⭐⭐⭐⭐⭐ | Encode categoricals correctly | Critical |
| ⭐⭐⭐⭐ | Log-transform skewed features | High |
| ⭐⭐⭐⭐ | Fit scaler on train only | High |
| ⭐⭐⭐⭐ | Create domain-informed ratio features | High |
| ⭐⭐⭐ | Date/time feature extraction | Medium–High |
| ⭐⭐⭐ | Remove redundant features | Medium |
| ⭐⭐ | Interaction features | Medium |
| ⭐⭐ | Cyclical encoding | Medium |
| ⭐ | Advanced text features (TF-IDF) | Situational |

Master the top 5 and you will outperform the vast majority of beginner practitioners.

---

*Feature Engineering Series | File 5 of 5 — Series Complete*

---

## 📚 Recommended Next Steps

1. **Practice**: Apply this pipeline to real datasets on Kaggle (Titanic, House Prices, Give Me Some Credit)
2. **Libraries to explore**: `feature-engine`, `featuretools`, `category_encoders`
3. **Advanced topics**: Automated feature engineering (AutoFeat), dimensionality reduction (PCA, t-SNE), embeddings for high-cardinality
4. **Read**: *Feature Engineering for Machine Learning* by Alice Zheng & Amanda Casari (O'Reilly)
