# Customer Churn Prediction - Bank Dataset Analysis

A data science project analyzing customer churn behavior using the **BankChurners** dataset. This notebook walks through the full pipeline: data cleaning, exploratory data analysis, statistical analysis, and a Decision Tree classification model to predict whether a customer will churn.

---

## Dataset

- **Source**: BankChurners.csv
- **Records**: 10,127 customers
- **Features**: Customer demographics, account activity, credit limits, transaction behavior, and more
- **Target**: `Attrition_Flag` — Existing Customer (0) vs. Attrited Customer (1)
- **Churn Rate**: 16.1% (1,627 churned customers) — imbalanced classification problem

---

## Project Pipeline

### 1. Data Cleaning
- Dropped irrelevant columns (`CLIENTNUM`, `Naive_Bayes` classifier columns)
- Converted binary text columns (`Attrition_Flag`, `Gender`) to numeric
- Handled "Unknown" categorical values in `Education_Level`, `Marital_Status`, and `Income_Category`

### 2. Exploratory Data Analysis (EDA)
| Visualization | Insight |
|---------------|---------|
| Churn distribution (pie) | Confirmed churn is a minority class (~16%) |
| Churn rate by income (bar) | U-shaped relationship — lowest and highest earners churn slightly more |
| Age distribution (histogram) | Heavy overlap between churned and retained customers — age is not a strong driver |
| Transaction count (boxplot) | Existing customers median ~70 transactions vs. ~40 for churned — strong signal |
| Spend vs. transactions (scatter) | Churned customers cluster in low-activity/low-spend corner |
| Correlation heatmap | `Total_Trans_Ct` (-0.37) and `Total_Revolving_Bal` (-0.25) most correlated with churn |

### 3. Statistical Analysis
Explicit calculation of **mean, median, mode, and standard deviation** for key features (`Customer_Age`, `Total_Trans_Ct`, `Total_Trans_Amt`, `Credit_Limit`) to confirm distributions and detect skew.

### 4. Feature Engineering
- Ordinal encoding for `Income_Category` and `Education_Level` (preserves natural order)
- One-hot encoding for `Marital_Status` and `Card_Category`
- Dropped `Avg_Open_To_Buy` due to near-perfect multicollinearity with `Credit_Limit`

### 5. Model: Decision Tree Classifier
- **Algorithm**: `DecisionTreeClassifier` (max_depth=5, random_state=42)
- **Train/Test Split**: 80/20, stratified by churn label

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 93.1% |
| Precision | 82.5% |
| Recall | 72.6% |
| F1 Score | 77.3% |

### Feature Importance (Top 3)
1. `Total_Trans_Ct` — **39%**
2. `Total_Trans_Amt` — **26%**
3. `Total_Revolving_Bal` — significant contributor

These findings directly align with EDA patterns, confirming transaction activity is the strongest churn predictor.

---

## Key Findings

- **Transaction activity is the strongest churn signal.** Customers with low transaction counts and low spend are far more likely to churn.
- **Revolving balance and relationship depth matter.** Low `Total_Revolving_Bal` and fewer bank products (`Total_Relationship_Count`) correlate with higher churn risk.
- **Age is not a meaningful factor** — distributions overlap heavily and the model assigns it <1% importance.
- **Income shows a mild U-shape** — both ends of the income spectrum churn slightly more than middle brackets.

---

## Recommendations

1. Use **transaction activity and revolving balance** as early warning indicators — flag customers whose activity drops for proactive retention outreach.
2. Since recall is the weaker metric, consider **adjusting the decision threshold** to trade false positives for catching more true churners.
3. Compare against other algorithms (Logistic Regression, Random Forest) and use cross-validation to tune `max_depth` more rigorously.

---

## Limitations

- Dataset is a snapshot — it cannot explain *why* transaction activity dropped before churn.
- ~10–15% of records had "Unknown" values for key categorical fields; kept as distinct categories rather than imputed.
- Model tuned for interpretability (max_depth=5); a deeper tree or ensemble method may improve recall.

---

## Files

| File | Description |
|------|-------------|
| `projectNotes.ipynb` | Full analysis notebook (cleaning, EDA, modeling, conclusions) |
| `BankChurners.csv` | Source dataset |
| `README.md` | Project overview |

---

## How to Run

1. Clone the repository
2. Ensure the following Python packages are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `projectNotes.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially

---


