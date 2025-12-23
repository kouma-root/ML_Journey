# Customer Churn Prediction Project

## Project Overview

This project aims to predict customer churn using machine learning techniques. The analysis uses a comprehensive dataset to build and compare multiple logistic regression models with different preprocessing strategies and feature selection approaches.

**Author:** Koumakang Parfait Aboyna

## Table of Contents

1. [Dataset Information](#dataset-information)
2. [Project Structure](#project-structure)
3. [Models Developed](#models-developed)
4. [Key Results](#key-results)
5. [How to Run](#how-to-run)
6. [Technologies Used](#technologies-used)

---

## Dataset Information

### Source
The dataset is loaded from `Churn.csv` containing customer information from a financial institution.

### Dataset Statistics
- **Total Records:** 10,000 customer records
- **Total Features (Initial):** 12 columns including customer demographics and account information
- **Target Variable:** Churn (Binary: 0 = No Churn, 1 = Churn)

### Features Description
| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numerical | Customer's credit score |
| Age | Numerical | Customer's age in years |
| Tenure | Numerical | Number of years as a customer |
| Balance | Numerical | Account balance |
| EstimatedSalary | Numerical | Customer's estimated annual salary |
| NumOfProducts | Numerical | Number of products customer uses |
| HasCrCard | Binary | Whether customer has a credit card |
| IsActiveMember | Binary | Whether customer is an active member |
| Gender | Categorical | Customer's gender |
| Geography | Categorical | Customer's location |
| Churn | Binary | Target variable (0 = No, 1 = Yes) |

### Data Preprocessing Steps
1. **Removed non-informative features:** RowNumber, CustomerId, Surname
2. **Encoded categorical variables:** Gender and Geography using LabelEncoder
3. **Created new derived features:**
   - BalanceSalaryRatio: Balance / (EstimatedSalary + 1)
   - TenureByAge: Tenure / (Age + 1)
   - CreditScoreGivenAge: CreditScore / (Age + 1)

### Churn Distribution
- **Non-Churned (0):** ~80% of customers
- **Churned (1):** ~20% of customers
- The data shows a class imbalance typical in churn prediction problems

---

## Project Structure

The project is organized into the following main phases:

### Phase 1: Exploratory Data Analysis (EDA)
- Dataset loading and initial inspection
- Statistical summaries and data shape analysis
- Distribution analysis of key features
- Correlation analysis with heatmaps

### Phase 2: Feature Engineering
- Categorical variable encoding
- Feature scaling using StandardScaler
- Creation of derived features
- Feature selection based on correlation threshold (0.1)

### Phase 3: Model Development
Six different logistic regression models were trained and evaluated:

1. **Model 1 (Baseline - All Features, Unscaled)**
2. **Model 2 (Baseline - All Features, Scaled)**
3. **Model 1 Reduced (Important Features Only, Unscaled)**
4. **Model 2 Reduced (Important Features Only, Unscaled)**
5. **Model 1 Scaled (Reduced Features, Scaled)**
6. **Model 2 Scaled (Reduced Features, Scaled)**

### Phase 4: Model Evaluation
- Accuracy calculation
- Confusion matrix analysis
- Classification reports (Precision, Recall, F1-Score)

---

## Models Developed

### Model Training Strategy

All models use **Logistic Regression**, a linear classification algorithm suitable for binary classification problems.

#### Feature Selection Approaches

**Approach 1 (important_features_1):**
Features selected from original numerical data with correlation >= 0.1 threshold

**Approach 2 (important_features_2):**
Features selected from engineered data (with new features) with correlation >= 0.1 threshold

#### Scaling Strategy

- **Unscaled:** Original feature values used directly
- **Scaled:** StandardScaler applied to normalize features to mean=0, std=1

### Model Configurations

| Model | Features | Scaling | Description |
|-------|----------|---------|-------------|
| Model 1 | All | No | Baseline with all original features |
| Model 2 | All | Yes | Baseline with scaled features |
| Model 1 Reduced | Selected | No | Reduced to important features only |
| Model 2 Reduced | Selected | No | Reduced with new features, unscaled |
| Model 1 Scaled | Selected | Yes | Reduced features with scaling |
| Model 2 Scaled | Selected | Yes | Reduced with new features and scaling |

---

## Key Results

### Performance Comparison Summary

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|---------------------|------------------|-------------------|
| **Model 1 (Baseline, Unscaled)** | 0.804 | 0.806 | 0.507 | 0.630 |
| **Model 2 (Baseline, Scaled)** | 0.804 | 0.806 | 0.507 | 0.630 |
| **Model 1 Reduced (Unscaled)** | 0.809 | 0.824 | 0.521 | 0.644 |
| **Model 2 Reduced (Unscaled)** | 0.815 | 0.821 | 0.546 | 0.658 |
| **Model 1 Scaled (Reduced)** | 0.809 | 0.824 | 0.521 | 0.644 |
| **Model 2 Scaled (Reduced+New Features)** | 0.815 | 0.821 | 0.546 | 0.658 |

### Key Findings

1. **Feature Selection Impact:** Models using selected features (threshold 0.1 correlation) consistently outperformed baseline models
2. **Feature Engineering Impact:** The addition of derived features improved model performance by 0.6-1.1% accuracy
3. **Scaling Impact:** StandardScaler showed minimal impact on Logistic Regression performance (as expected for this algorithm)
4. **Best Model:** Model 2 Scaled (Reduced+New Features) achieved the highest accuracy of 81.5% with better recall for positive class

### Visualizations Generated

- Churn distribution countplot
- Age distribution histogram
- Credit score distribution histogram
- Age vs Churn boxplot
- Balance vs Churn boxplot
- Credit score vs Churn boxplot
- Geography vs Churn countplot
- Gender vs Churn countplot
- Correlation heatmaps (before and after feature engineering)

---

## How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Execution Steps

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/project
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook ML_Project_Churn.ipynb
   ```

3. **Run cells in order:**
   - Step 1: Load Dataset and Libraries
   - Step 2: Dataset Overview
   - Steps 3-9: Exploratory Data Analysis
   - Steps 10-14: Data Preprocessing and Feature Engineering
   - Steps 15-16: Model Training and Evaluation

### Expected Output
- Statistical summaries
- Visualization plots
- Model accuracy scores
- Confusion matrices
- Classification reports

---

## Technologies Used

### Libraries
- **NumPy:** Numerical computing and array operations
- **Pandas:** Data manipulation and analysis
- **Matplotlib:** Data visualization
- **Seaborn:** Advanced statistical visualization
- **Scikit-learn:** Machine learning algorithms and metrics

### Algorithms
- **Logistic Regression:** Binary classification model
- **StandardScaler:** Feature normalization
- **LabelEncoder:** Categorical variable encoding
- **Train-Test Split:** Data partitioning (80-20 split)

---

## Recommendations for Improvement

1. **Hyperparameter Tuning:** Experiment with different C values and regularization types
2. **Advanced Algorithms:** Test Random Forest, Gradient Boosting, or Neural Networks
3. **Class Imbalance:** Consider SMOTE, class weights, or oversampling techniques
4. **Cross-Validation:** Use k-fold cross-validation for more robust evaluation
5. **Additional Features:** Engineer domain-specific features based on business knowledge
6. **Ensemble Methods:** Combine multiple models for improved predictions
7. **Threshold Optimization:** Adjust decision threshold based on business costs

---

## Project Files

- `ML_Project_Churn.ipynb` - Main Jupyter notebook with full analysis
- `Churn.csv` - Customer dataset
- `README.md` - This file
- `DOCUMENTATION.md` - Detailed technical documentation

---

## Author Notes

This project demonstrates a complete machine learning pipeline from data exploration to model evaluation. The iterative approach of feature selection and engineering shows how careful data preprocessing can improve model performance beyond baseline approaches.

For questions or improvements, please refer to the detailed documentation.
