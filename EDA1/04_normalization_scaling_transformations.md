# 📕 File 4: Normalization, Scaling & Transformations

> **Series:** Feature Engineering for Data Science — 80/20 Guide  
> **Level:** Intermediate  
> **Prerequisite:** Files 1–3 (Data Types, EDA, Encoding)

---

## 🧭 Why Scale and Normalize?

Many machine learning algorithms are **sensitive to the scale of features**. Consider salary (range: $20,000–$200,000) and age (range: 18–85). Without scaling:

- **Distance-based models** (KNN, K-Means, SVM): Salary dominates because its raw values are 1000× larger
- **Gradient descent** (Linear Regression, Neural Networks): Converges very slowly when features have different scales
- **Regularization** (Lasso, Ridge): Penalizes large-coefficient features, which depend on feature scale

**Tree-based models** (Decision Tree, Random Forest, XGBoost) split on thresholds and are **scale-invariant** — they do not require scaling.

---

## 🗺️ Scaling Decision Map

```
What model are you using?
│
├── Tree-based (RF, XGBoost, Decision Tree) → Skip scaling (optional)
│
└── Everything else → Scale
         │
         ├── Does your data have many outliers?
         │    ├── YES → Robust Scaler
         │    └── NO  → Does your data need to be normally distributed?
         │                  ├── YES → PowerTransformer or QuantileTransformer
         │                  └── NO  → Need values between 0 and 1?
         │                                ├── YES → Min-Max Scaler
         │                                └── NO  → Standard Scaler
         │
         └── Is the feature heavily right-skewed (income, price)?
              └── Log Transform first, then Standard Scale
```

---

## 1. Standard Scaler (Z-Score Normalization)

Transforms features to have **mean = 0** and **standard deviation = 1**.

```
z = (x - μ) / σ
```

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

salary = np.array([30000, 50000, 80000, 120000, 200000]).reshape(-1, 1)

scaler = StandardScaler()
salary_scaled = scaler.fit_transform(salary)

print(salary_scaled.flatten().round(2))
# [-1.26, -0.72, -0.05,  0.55,  1.48]
# mean ≈ 0, std ≈ 1

print(f"Mean: {salary_scaled.mean():.4f}, Std: {salary_scaled.std():.4f}")
# Mean: 0.0000, Std: 1.0000
```

**When to use:**
- Logistic Regression, Linear Regression, SVM, KNN, Neural Networks
- When data is **approximately normally distributed**
- When outliers are not extreme

**Limitation:** Affected by outliers — a single extreme value stretches the scale.

---

## 2. Min-Max Scaler (Normalization)

Scales features to a fixed range, typically **[0, 1]**.

```
x_scaled = (x - x_min) / (x_max - x_min)
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))  # default
salary_scaled = scaler.fit_transform(salary)

print(salary_scaled.flatten().round(2))
# [0.  , 0.12, 0.29, 0.53, 1.  ]
```

**When to use:**
- Neural networks (especially when activation functions expect [0,1] or [-1,1])
- Image pixel values (divide by 255)
- When you need a bounded output

**Critical limitation:** Very sensitive to outliers. One outlier at $10,000,000 would compress all salaries near 0.

```python
# Demonstrating the outlier problem
salary_with_outlier = np.array([30000, 50000, 80000, 120000, 200000, 10000000])
salary_with_outlier_scaled = MinMaxScaler().fit_transform(
    salary_with_outlier.reshape(-1, 1)
)
print(salary_with_outlier_scaled.flatten().round(4))
# [0.002, 0.004, 0.007, 0.011, 0.019, 1.0   ]
# All real salaries crushed near 0!
```

---

## 3. Robust Scaler

Uses the **median and IQR** instead of mean and std — resistant to outliers.

```
x_scaled = (x - median) / IQR
```

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
salary_scaled = scaler.fit_transform(salary)

print(salary_scaled.flatten().round(2))
# Outlier-resistant scaling
```

**When to use:**
- Data has significant outliers that you don't want to remove
- Medical data, financial data, sensor data

**Note:** The output is NOT bounded between 0 and 1. It centers around 0 but outliers will still be far from zero — they just don't distort the scaling of other points.

---

## 4. Log Transformation

Compresses the range of **right-skewed** distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

income = np.array([20000, 25000, 30000, 35000, 45000, 80000, 250000, 1000000])

income_log = np.log1p(income)  # log(1 + x) — safe for x=0

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(income, bins=20, color='salmon')
axes[0].set_title('Original Income — Right Skewed')
axes[1].hist(income_log, bins=20, color='steelblue')
axes[1].set_title('Log-Transformed Income — More Normal')
plt.tight_layout()
plt.show()
```

**Why `np.log1p` instead of `np.log`?**
`log(0)` is undefined (negative infinity). `log1p(x) = log(1+x)` handles zero safely.

**When to use:**
- Monetary values: salary, house price, revenue
- Counts: page views, number of transactions
- Any heavily right-skewed distribution

**Requirements:**
- All values must be ≥ 0 (use `log1p` for zeros)
- Should be applied **before** other scaling

**Checking if log transform helps:**

```python
import scipy.stats as stats

# Check skewness
print(f"Skewness before: {stats.skew(income):.2f}")    # e.g., 3.21
print(f"Skewness after:  {stats.skew(income_log):.2f}") # e.g., 0.74

# Visual check
from scipy.stats import probplot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
probplot(income, plot=axes[0])
axes[0].set_title('Q-Q Plot: Original')
probplot(income_log, plot=axes[1])
axes[1].set_title('Q-Q Plot: Log-Transformed')
plt.tight_layout()
plt.show()
```

A good Q-Q plot will have points close to the diagonal line.

---

## 5. Power Transformer (Box-Cox and Yeo-Johnson)

More powerful than log — finds the **optimal power transformation** to make data as normal as possible.

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson: works with negative values too
pt_yj = PowerTransformer(method='yeo-johnson')
income_transformed = pt_yj.fit_transform(income.reshape(-1, 1))

# Box-Cox: only for strictly positive data
pt_bc = PowerTransformer(method='box-cox')
income_transformed_bc = pt_bc.fit_transform(income.reshape(-1, 1))

print(f"Optimal lambda (Box-Cox): {pt_bc.lambdas_[0]:.3f}")
```

**What it does internally:**
```
Box-Cox:
  x_new = (x^λ - 1) / λ   if λ ≠ 0
  x_new = log(x)           if λ = 0

Yeo-Johnson extends this to support negative values.
```

**When to use:**
- When log transform isn't enough
- Linear models that assume normality of features
- When you want automated optimization

---

## 6. Quantile Transformer

Maps any distribution to a **uniform** or **normal** distribution using empirical quantiles.

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
income_qt = qt.fit_transform(income.reshape(-1, 1))
```

**When to use:**
- Most extreme transformation — use when data is very non-normal
- Robust to outliers (maps all quantiles proportionally)
- Works on any distribution shape

**Limitation:** Distorts the original distribution more aggressively. Loses interpretability.

---

## 7. The Golden Rule: Fit on Train, Transform on Both

**Never fit your scaler on the test set.** This is **data leakage**.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# ✅ CORRECT
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform on train
X_test_scaled  = scaler.transform(X_test)         # only transform on test

# ❌ WRONG
X_test_wrong = scaler.fit_transform(X_test)  # refitting on test leaks test stats into training
```

**Use scikit-learn Pipelines to enforce this automatically:**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression())
])

pipeline.fit(X_train, y_train)   # scaler.fit only sees X_train
pipeline.score(X_test, y_test)   # scaler.transform applied correctly
```

---

## 8. Comparing All Scalers Side by Side

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer

X = np.array([10, 12, 14, 15, 16, 18, 20, 25, 100, 500]).reshape(-1, 1)

scalers = {
    'Original':            X,
    'Standard Scaler':     StandardScaler().fit_transform(X),
    'MinMax Scaler':       MinMaxScaler().fit_transform(X),
    'Robust Scaler':       RobustScaler().fit_transform(X),
    'PowerTransformer':    PowerTransformer(method='yeo-johnson').fit_transform(X),
    'QuantileTransformer': QuantileTransformer(output_distribution='normal', n_quantiles=10).fit_transform(X),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, data) in zip(axes.flatten(), scalers.items()):
    ax.hist(data.flatten(), bins=10, color='steelblue', edgecolor='black')
    ax.set_title(name)
    ax.set_xlabel('Value')
plt.suptitle('Effect of Different Scaling Methods', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 📋 Scaling Quick Reference

| Scaler | Formula | Output Range | Outlier Robust | Best For |
|--------|---------|-------------|---------------|----------|
| Standard | z = (x-μ)/σ | Unbounded | No | Most linear models |
| MinMax | (x-min)/(max-min) | [0, 1] | No | Neural networks |
| Robust | (x-median)/IQR | Unbounded | **Yes** | Data with outliers |
| Log | log(1+x) | Unbounded | Partial | Right-skewed, positive |
| PowerTransformer | Box-Cox/Yeo-Johnson | Unbounded | Partial | Near-normal output needed |
| QuantileTransformer | Empirical quantiles | [0,1] or N(0,1) | **Yes** | Any distribution |

---

## 📋 Scaling Checklist

- [ ] Identify which model you are using (tree-based → scaling optional)
- [ ] Check distribution of each numeric feature (histogram, skewness)
- [ ] Apply log transform to right-skewed features first
- [ ] Select appropriate scaler based on model + distribution
- [ ] Fit scaler **only on training data**
- [ ] Use Pipelines to prevent leakage
- [ ] Re-check distribution after transformation (Q-Q plot)

---

## 📚 What's Next?

In **File 5**, we cover **Feature Creation & Feature Selection** — how to create new informative features from existing ones, and how to prune away features that hurt model performance.

---

*Feature Engineering Series | File 4 of 5*
