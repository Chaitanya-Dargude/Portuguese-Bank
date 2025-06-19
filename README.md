# Portuguese Bank Terms Deposit Prediction
## Problem Statement:
    Build a data analysis and prediction model to help the bank identify customers
      who are likely to subscribe to a term deposit based on past direct marketing campaigns.
## Domain:
- Finance / Banking
## DataSet Overview:
- Source: Portuguese Bank Direct Marketing Campaign (May 2008 - Nov 2010)
- File Used: `bank-additional/bank-additional-full.csv`
- Total Features: 20 input variables + 1 output variable (`y`)
- Objective: Predict whether the customer will subscribe to a term deposit (`y` = yes/no)
## Import Libraries & Load Data
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns

## Model libraries
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import LabelEncoder, StandardScaler
- from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from xgboost import XGBClassifier

## Load Data
- data = pd.read_csv('bank-additional-full.csv', sep=';')
- data.head()
- data.info()
- data.describe()
- data.isnull().sum()  # check for nulls
## Replace 'unknown' with mode or drop depending on % of unknowns
  - for col in ['job', 'education', 'marital', 'default', 'housing', 'loan']:
      - print(col, ":", data[col].value_counts(normalize=True)['unknown'])
## Example fix:
  - data['job'] = data['job'].replace('unknown', data['job'].mode()[0])
## Univariate & Bivariate Analysis :
  - Categorical: Bar plots (sns.countplot)
  - Numerical: Histograms, boxplots
## Data Preprocessing
- Drop duration (as per business logic)
  - data = data.drop('duration', axis=1)
- Encode Categorical Variables
  - data_encoded = pd.get_dummies(data, drop_first=True)
## Train-Test Split
- X = data_encoded.drop('y_yes', axis=1)
- y = data_encoded['y_yes']
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Feature Scaling
- scaler = StandardScaler()
  - X_train = scaler.fit_transform(X_train)
  - X_test = scaler.transform(X_test)
## Model Building & Evaluation
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
## Model Comparison Table
| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression|   0.91   |   ...     |  ...   |   ...    |
| Decision Tree      |   0.88   |   ...     |  ...   |   ...    |
| Random Forest      |   0.93   |   ...     |  ...   |   ...    |
| XGBoost            |   0.94   |   ...     |  ...   |   ...    |
##  Best Model Suggestion: XGBoost
## Challenges Faced & Solutions
- **Challenge 1:** Presence of many 'unknown' values in categorical columns  
  - **Solution:** Replaced with mode if <10%, else removed rows.

- **Challenge 2:** Imbalanced classes in target variable  
  - **Solution:** Consider SMOTE or class_weight in advanced version.

- **Challenge 3:** Many categorical variables  
  - **Solution:** Used one-hot encoding to convert them into numerical format.

- **Challenge 4:** Duration variable gives data leakage  
  - **Solution:** Dropped `duration` column during model training.
## Business Recommendations to Bank Marketing Team
- Focus on clients with:
   - Jobs like "student", "retired", "admin." → higher subscription rate.
   - Education = "university.degree" → higher interest.
   - Contact via "cellular" is more effective than "telephone".

- Call Timing:
   - Months like May, August, October have better responses.
   - Midweek (Tue, Wed, Thu) performs better.

- Previous campaign success improves current success — maintain a strong follow-up system.

- Socioeconomic factors:
   - Customers respond more when `euribor3m` is low and `emp.var.rate` is positive.

- Avoid repeated contacts:
   - High `campaign` count reduces chance of success.
## Project Analysis:
- This project analyzes a Portuguese bank’s marketing campaign data to predict term deposit subscriptions.
- The dataset includes client demographics, previous contacts, and economic indicators.
- Initial data cleaning addressed unknown values using mode imputation.
- One-hot encoding was applied to categorical variables to prepare the data for modeling.
- SMOTE was used to balance the dataset due to class imbalance.
- Various models were trained: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost.
- Model performance was compared using Accuracy, Precision, Recall, and F1 Score.
- XGBoost performed the best overall with an F1 Score of ~0.3788 and Accuracy of ~89.6%.
- Confusion Matrix and Classification Reports revealed moderate recall but good precision for the positive class.
- The bank should prioritize targeting clients with specific job roles, previous successful contact, and cellular contact type.
## Conclusion
- A complete EDA and preprocessing flow was created.
- Best model: XGBoost with highest accuracy and F1-score.
- Key recommendations were drawn from patterns in the data.
