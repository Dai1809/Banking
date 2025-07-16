# 💳 Credit Card Fraud Detection

This project detects fraudulent transactions using various machine learning classification models.

## 📂 Dataset
- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Note:** File `creditcard.csv` is excluded from the repository due to GitHub’s 100MB file size limit.

## 🛠️ Features
- Highly imbalanced dataset (fraudulent cases ~0.17%)
- Time and Amount are scaled using `StandardScaler`
- Models used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## 📉 Performance
Model evaluation focuses on **Recall** and **Precision** due to class imbalance.

## 📦 Libraries Used
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib

## 🚫 Large Files
To avoid GitHub push errors, the `creditcard.csv` file is excluded via `.gitignore`.
