import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


train_df = pd.read_csv('train_loan.csv')
print(train_df.shape)
print(train_df.head())

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['object', 'category']:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
               
            elif df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
               
    return df

train_df = fill_missing_values(train_df)

# Drop unnecassary field
train_df = train_df.drop('Loan_ID',axis=1)

# One Hot Encoding done here
train_df = pd.get_dummies(train_df, drop_first=True)

# Plan the output and feature 
x = train_df.drop('LoanAmount',axis=1)
y = train_df['LoanAmount']

#Split training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Applying Linear Regression 
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Predict using linear regression
lr_pred = lr_model.predict(x_test)


tree = DecisionTreeRegressor(random_state=40)
tree.fit(x_train, y_train)
tree_pred = tree.predict(x_test)


forest = RandomForestRegressor(n_estimators=100, random_state=40)
forest.fit(x_train, y_train)
forest_pred = forest.predict(x_test)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=40)
xgb_model.fit(x_train, y_train)

# Predict
xgb_pred = xgb_model.predict(x_test)


def modelevaluation (Model,y_actual,y_pred):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'{Model} scores : MAE :{mae} ,MSE: {rmse} , R2: {r2}   ')

modelevaluation('Linear Regression',y_test,lr_pred)
modelevaluation('DecissionTree Regression',y_test,tree_pred)
modelevaluation('RandomForest Regression',y_test,forest_pred)
modelevaluation('XGB Regression',y_test,xgb_pred)




# Collect model names and metrics
models = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]

mae_vals = [
    mean_absolute_error(y_test, lr_pred),
    mean_absolute_error(y_test, tree_pred),
    mean_absolute_error(y_test, forest_pred),
    mean_absolute_error(y_test, xgb_pred)
]

rmse_vals = [
    np.sqrt(mean_squared_error(y_test, lr_pred)),
    np.sqrt(mean_squared_error(y_test, tree_pred)),
    np.sqrt(mean_squared_error(y_test, forest_pred)),
    np.sqrt(mean_squared_error(y_test, xgb_pred))
]

r2_vals = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, tree_pred),
    r2_score(y_test, forest_pred),
    r2_score(y_test, xgb_pred)
]

# Create subplots
plt.figure(figsize=(16, 5))

# MAE Plot
plt.subplot(1, 3, 1)
plt.bar(models, mae_vals, color='skyblue')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.xticks(rotation=15)

# RMSE Plot
plt.subplot(1, 3, 2)
plt.bar(models, rmse_vals, color='lightgreen')
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')
plt.xticks(rotation=15)

# R² Score Plot
plt.subplot(1, 3, 3)
plt.bar(models, r2_vals, color='salmon')
plt.title('R² Score')
plt.ylabel('R²')
plt.xticks(rotation=15)

plt.suptitle('Model Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
