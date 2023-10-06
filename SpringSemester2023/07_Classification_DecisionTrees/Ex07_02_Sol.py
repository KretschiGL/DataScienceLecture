# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex07_02_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Creating Labels and Features"))
labels = ["No", "Yes"]
features = data.columns.drop(["Loan_ID", "LoanStatus"]).values
display(labels)
display(features)

display(Markdown("##### Creating Train & Test sets"))
X = data[features]
y = data["LoanStatus"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=42)
display(f"Train set size: {len(X_train)}")
display(f"Test set size: {len(X_test)}")

display(Markdown("##### Creating, training and using the model"))
model = DecisionTreeClassifier(min_samples_leaf=24)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(accuracy_score(y_test, y_pred))

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(xlabel="True Labels", ylabel="Predicted Labels")
plt.show()

display(Markdown("##### Plotting the Tree"))
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, ax=ax, filled=True, rounded=True, feature_names=features, class_names=labels, fontsize=10)
plt.show()
display(Markdown("##### The Problem"))
display(Markdown("The decision is based on the credit history. Either there exists one or not. After that decision, classes won't change."))

display(Markdown("##### Loading new Data"))
data_use = pd.read_csv("./Ex07_02_Data_Use.csv", sep=";")
display(data_use.head(5))

display(Markdown("##### Predicting the Probabilities"))
y_proba = model.predict_proba(data_use.drop("Loan_ID", axis="columns"))
display(y_proba[:5])

display(Markdown("##### Converting Probabilities to DataFrame"))
loans = pd.DataFrame(y_proba, columns=["No", "Yes"])[["Yes", "No"]]
display(loans.head(5))

display(Markdown("##### Combining Probabilities with Data"))
loan_proba = pd.concat([data_use["Loan_ID"], loans, data_use.drop("Loan_ID", axis="columns")], axis="columns")
display(loan_proba.head(5))