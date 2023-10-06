# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

display(Markdown("###### Loading Telecom Customer Data"))
data = pd.read_csv("./Ex14_Class-02_Data-Train.csv", sep=";")
display(data.head(5))
display(data.info())

display(Markdown("###### Preprocessing"))
data = data.drop("CustomerID", axis="columns")
data = data.replace({"Yes":1,"No":0})
data["MaritalStatus"] = data["MaritalStatus"].replace({0:"No", 1:"Yes"})
data["CreditRating"] = data["CreditRating"].apply(lambda r: r.split("-")[0]).astype(np.int64)
data = pd.get_dummies(data)
display(data.head(5))
display(data.info())

display(Markdown("###### Random Forest Training (with default values)"))
display(Markdown("This may take a while..."))
X = data.drop("Churn", axis="columns")
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
model = RandomForestClassifier(max_samples=.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

display(Markdown("###### Optimizing Random Forest Classifier"))
display(Markdown("This may take a while..."))
params = {
    "max_features":["auto", "log2"],
    "max_depth": [4,6,8, None],
    "min_samples_split":[2,10,20,50]
}
display(params)
grid = GridSearchCV(model, params, cv=5, n_jobs=4)
grid.fit(X, y)
display(Markdown(f"Best model score: {grid.best_score_}"))
model_best = grid.best_estimator_
display(model_best)
model_best.fit(X_train, y_train)
y_pred = model_best.predict(X_test)
display(f"Accuracy score (best model): {accuracy_score(y_test, y_pred)}")

display(Markdown("###### Retraining Best Model"))
model_best.fit(X, y)
display(model_best)

display(Markdown("###### Predicting Churn for Holdout Data"))
data_holdout = pd.read_csv("./Ex14_Class-02_Data-Holdout.csv", sep=";")
data_holdout = data_holdout.replace({"Yes":1,"No":0})
data_holdout["MaritalStatus"] = data_holdout["MaritalStatus"].replace({0:"No", 1:"Yes"})
data_holdout["CreditRating"] = data_holdout["CreditRating"].apply(lambda r: r.split("-")[0]).astype(np.int64)
data_holdout = pd.get_dummies(data_holdout)
data_churn = pd.DataFrame(model_best.predict_proba(data_holdout.drop("CustomerID",axis="columns")))
data_churn = data_churn.rename(columns={0:"Stays", 1:"Churn"})
data_churn = pd.concat([data_churn[["Churn", "Stays"]], data_holdout], axis="columns")
data_churn = data_churn.sort_values("Churn", ascending=False)
display(data_churn.head(5))
