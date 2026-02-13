# 🏠 House Price Prediction using Regularized Linear Models

## 📌 Project Overview

This project builds an end-to-end Machine Learning regression pipeline to predict house prices using:

- Linear Regression
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)

The objective was to understand the impact of regularization, handle skewed targets, and implement a structured ML workflow from EDA to model evaluation.

---

## 📊 Exploratory Data Analysis (EDA)

- Inspected dataset structure and feature types
- Analyzed missing values and percentage distribution
- Generated correlation heatmap
- Identified features highly correlated with `SalePrice`
- Visualized key relationships using scatter plots and boxplots

---

## 🔄 Target Transformation

- Initial target skewness ≈ **1.88** (positively skewed)
- Applied **log transformation**
- Reduced skewness to ≈ **0.12**
- Improved normality assumption for regression models

---

## 🧹 Data Preprocessing

- Dropped columns with >80% missing values
- Applied appropriate imputation:
  - Categorical → `'None'`
  - Numerical → `0` / median
  - Mode for specific features
- Performed One-Hot Encoding (`drop_first=True`)
- Split data into **80% training** and **20% testing**
- Applied **StandardScaler** before regularization

---

## 🤖 Model Training & Evaluation

### 1️⃣ Linear Regression
Baseline model performance evaluated using:
- R² Score
- Mean Squared Error (MSE)

### 2️⃣ Ridge Regression (L2)
- Cross-validation used to select optimal alpha
- Reduced multicollinearity impact
- Improved generalization

### 3️⃣ Lasso Regression (L1)
- Automatic alpha selection via LassoCV
- Performed feature shrinkage
- Simplified model by reducing coefficient magnitudes

---

## 📈 Model Comparison

| Model               | R² Score (Approx.) |
|---------------------|-------------------|
| Linear Regression   | ~0.87 |
| Ridge Regression    | ~0.88 |
| Lasso Regression    | ~0.83 |

**Ridge Regression performed slightly better due to effective regularization.**

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🎯 Key Learnings

- Importance of handling skewed targets
- Effect of regularization on overfitting
- Why feature scaling is necessary for penalized models
- Structured end-to-end ML workflow implementation

---

## 🚀 Future Improvements

- Implement Pipeline for cleaner workflow
- Add K-Fold Cross Validation scoring
- Visualize feature importance
- Deploy model using Flask or Streamlit

---

## 👤 Author

**Anshu Pramanik**  
 Aspiring Data Scientist



