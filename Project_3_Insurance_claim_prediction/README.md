# Insurance Claim Prediction

A machine learning project to predict whether a building will have insurance claims based on property characteristics. This project demonstrates the complete ML pipeline, from data exploration to model evaluation and interpretation.

## 📌 Project Overview

Insurance companies need to assess risk when determining policy pricing and terms. This project builds predictive models to identify which properties are likely to have claims filed during their insurance period. By analyzing building characteristics like location, size, structure, and maintenance status, we can help insurers make data-driven decisions.

**Target Variable:** Whether a property had at least one claim filed during the insurance period (Yes/No)

## 🎯 Problem Statement

Given various property features (building type, location, painted status, fence condition, etc.), predict if the property will have an insurance claim within the insured period. This is a **binary classification problem**.

## 📊 Dataset

The dataset contains information about insured properties with the following characteristics:

| Feature | Description |
|---------|-------------|
| **Customer Id** | Unique identifier for the policyholder |
| **YearOfObservation** | Year of data collection |
| **Insured_Period** | Duration of insurance coverage (e.g., 1 = full year, 0.5 = 6 months) |
| **Residential** | Whether the building is residential |
| **Building_Painted** | Whether the building is painted (N=Painted, V=Not Painted) |
| **Building_Fenced** | Whether the building is fenced (N=Fenced, V=Not Fenced) |
| **Garden** | Whether the building has a garden (V=Yes, O=No) |
| **Settlement** | Area type (R=Rural, U=Urban) |
| **Building Dimension** | Size of the building in square meters |
| **Building_Type** | Category of building (Type 1, 2, 3, or 4) |
| **Date_of_Occupancy** | When the building was first occupied |
| **NumberOfWindows** | Count of windows in the building |
| **Geo Code** | Geographical location code |
| **Claim** | **Target** - Whether a claim was filed (0=No, 1=Yes) |

## 📁 Project Structure

```
Project_3_Insurance_claim_prediction/
├── README.md                          # This file
├── data/                              # Processed and raw datasets
│   ├── Train_data.csv                 # Original training dataset
│   ├── cleaned_train_data.csv         # Data after cleaning
│   ├── X_train_processed.csv          # Processed features for training
│   ├── X_val_processed.csv            # Processed features for validation
│   ├── y_train.csv                    # Training targets
│   ├── y_val.csv                      # Validation targets
│   ├── Variable_Description.csv       # Data dictionary
│   ├── logistic_regression_coefficients.csv    # Model weights
│   └── random_forest_feature_importances.csv   # Feature importance
└── notebooks/                         # Jupyter notebooks documenting the workflow
    ├── 01_data_overview.ipynb         # Initial data exploration
    ├── 02_data_quality.ipynb          # Quality assessment and diagnostics
    ├── 03_data_cleaning.ipynb         # Handling missing values and duplicates
    ├── 04_eda.ipynb                   # Exploratory Data Analysis
    ├── 05_feature_engineering.ipynb   # Creating and transforming features
    ├── 06_modeling.ipynb              # Training Logistic Regression and Random Forest
    └── 07_evaluation_and_explainability.ipynb  # Model evaluation and interpretation
```

## 🔄 Workflow Pipeline

The project follows a structured data science workflow:

### 1. **Data Overview** (`01_data_overview.ipynb`)
   - Load the raw dataset
   - Initial shape and structure exploration
   - Basic statistics and data types inspection

### 2. **Data Quality** (`02_data_quality.ipynb`)
   - Identify missing values and patterns
   - Check for data inconsistencies
   - Assess data completeness and quality metrics

### 3. **Data Cleaning** (`03_data_cleaning.ipynb`)
   - Remove duplicate rows
   - Handle missing values:
     - *Numerical columns:* Use median imputation
     - *Categorical columns:* Replace with "Unknown"
   - Prepare clean dataset for analysis

### 4. **Exploratory Data Analysis** (`04_eda.ipynb`)
   - Visualize distributions of features
   - Analyze relationships between variables
   - Understand target variable balance
   - Identify patterns and outliers

### 5. **Feature Engineering** (`05_feature_engineering.ipynb`)
   - Encode categorical variables
   - Scale numerical features
   - Create derived features if needed
   - Prepare final feature set for modeling

### 6. **Modeling** (`06_modeling.ipynb`)
   - Train **Logistic Regression** model
   - Train **Random Forest Classifier** model
   - Compare model performance
   - Select best performing model

### 7. **Evaluation & Explainability** (`07_evaluation_and_explainability.ipynb`)
   - Generate ROC curves and AUC scores
   - Analyze feature importance
   - Interpret model predictions
   - Provide business insights

## 🛠️ Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and metrics
- **Matplotlib** - Data visualization
- **Jupyter Notebooks** - Interactive documentation and code execution

## 📈 Models Implemented

### 1. Logistic Regression
- **Advantage:** Interpretable, fast to train, probabilistic outputs
- **Use case:** Baseline model with clear coefficient interpretation
- **Output:** Probability scores and feature coefficients

### 2. Random Forest Classifier
- **Advantage:** Handles non-linear relationships, feature importance, robust
- **Use case:** More complex pattern detection
- **Output:** Feature importance rankings for all variables

## 🚀 Getting Started

### Prerequisites
Ensure you have Python and required libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Running the Project

1. **Navigate to the project directory:**
   ```bash
   cd Project_3_Insurance_claim_prediction
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Execute notebooks in order:**
   - Start with `01_data_overview.ipynb`
   - Progress through each notebook sequentially
   - Each notebook builds on the previous one's outputs

4. **View results:**
   - Check `/data/` folder for processed datasets
   - Review generated coefficients and importance files
   - Analyze visualizations and metrics in notebook outputs

## 💡 Key Insights

After completing the analysis, the models reveal:

- **Feature Importance:** Which property characteristics most strongly predict claims
- **Risk Factors:** Building conditions and locations associated with higher claim rates
- **Model Performance:** Comparison of Logistic Regression vs Random Forest accuracy
- **Actionable Insights:** Building attributes insurers should prioritize in risk assessment

## 📊 Expected Outputs

- Processed training and validation datasets
- Trained model objects and coefficients
- Performance metrics and ROC curves
- Feature importance rankings
- Visual comparisons of model performance

## 📝 Notes for Beginners

- **Binary Classification:** This is a "yes/no" prediction problem (claim or no claim)
- **Train-Validation Split:** Data is split to train models and test on unseen data
- **Imbalanced Data:** Pay attention to the distribution of claims vs non-claims
- **Model Interpretability:** Both models offer different ways to understand predictions
  - Logistic Regression: Direct coefficient interpretation
  - Random Forest: Feature importance scores

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different preprocessing techniques
- Try additional models (SVM, Gradient Boosting, etc.)
- Create new features to improve predictions
- Optimize hyperparameters for better performance

## 📚 Learning Resources

To deepen your understanding of concepts used in this project:
- **Classification:** Understanding binary classification and metrics
- **Preprocessing:** Data cleaning and feature scaling techniques
- **Models:** How Logistic Regression and Random Forest work
- **Evaluation:** Interpreting AUC, ROC curves, and other metrics



**Developed by Koumakang Parfait Aboyna**  🎓
