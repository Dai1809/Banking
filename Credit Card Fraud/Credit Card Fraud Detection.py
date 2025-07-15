import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt


ds=pd.read_csv('creditcard.csv')


def fillmissingvalues(ds):
    for col in ds.columns :
        if ds[col].isnull().sum() > 0 :
            if ds[col].dtype in ['object','category']:
                modeval = ds[col].mode()[0]
                ds[col] = ds[col].fillna(modeval)

            elif ds[col].dtype in ['int64', 'float64']:
                modeval = ds[col].median()
                ds[col] = ds[col].fillna(modeval)    
    return ds


# fill missing values 

cleands = fillmissingvalues(ds)

## Scaling



scaler = StandardScaler()
ds['Amount'] = scaler.fit_transform(ds[['Amount']])
ds['Time'] = scaler.fit_transform(ds[['Time']])


# one hot encoding 

onehotcleands = pd.get_dummies (cleands , drop_first=True)


## Dropping the output from dataset 
y = onehotcleands['Class']
x = onehotcleands.drop('Class', axis=1)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=40,stratify=y)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train,y_train)
log_pred = log_reg.predict(x_test)

d_class = DecisionTreeClassifier()
d_class.fit(x_train,y_train)
d_pred = d_class.predict(x_test)

rf_class = RandomForestClassifier(n_estimators=50, random_state=40)
rf_class.fit(x_train,y_train)
rf_pred = rf_class.predict(x_test)


x_class = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
x_class.fit(x_train,y_train)
x_pred = x_class.predict(x_test)



def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }


results = []
results.append(evaluate_model("Logistic Regression", y_test, log_pred))
results.append(evaluate_model("Decision Tree", y_test, d_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))
results.append(evaluate_model("XGBoost", y_test, x_pred))

# Convert to DataFrame for plotting
results_df = pd.DataFrame(results)

# Plot each metric
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
plt.figure(figsize=(15, 6))

for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    plt.bar(results_df['model'], results_df[metric], color='skyblue')
    plt.title(metric.capitalize())
    plt.xticks(rotation=15)
    plt.ylim(0, 1)

plt.suptitle("📊 Model Evaluation Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
