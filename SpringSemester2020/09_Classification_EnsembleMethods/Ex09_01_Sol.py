# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex09_01_Data.csv")
display(data.head(5))

display(Markdown("##### Class Imbalance"))
fig, ax = plt.subplots(figsize=(5,5))
data["LoanStatus"].value_counts().rename({0:"Denied", 1:"Accepted"}).plot.bar(ax=ax, rot=0)
plt.show()

display(Markdown("##### Fixing the Class Imbalance"))
loans_denied = data[data["LoanStatus"] == 0].sample(400, replace=True, random_state=42)
loans_accepted = data[data["LoanStatus"] == 1]
loan_data = pd.concat([loans_accepted, loans_denied])
display(Markdown(f"Size: {len(loan_data)}"))
display(loan_data.head(5))
fig, ax = plt.subplots(figsize=(5,5))
loan_data["LoanStatus"].value_counts().rename({0:"Denied", 1:"Accepted"}).plot.bar(ax=ax, rot=0)
plt.show()

display(Markdown("##### Creating Train & Test sets"))
X = loan_data.drop(["Loan_ID", "LoanStatus"], axis=1)
y = loan_data["LoanStatus"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
display(Markdown(f"Train set size: {len(X_train)}"))
display(Markdown(f"Test set size: {len(X_test)}"))

display(Markdown("##### Labels & Features"))
labels = ["Denied", "Accepted"]
features = loan_data.columns.drop(["Loan_ID", "LoanStatus"]).values
display(Markdown(f"Labels: {labels}"))
display(Markdown(f"Features: {features}"))

display(Markdown("##### Training the Random Forest"))
model = RandomForestClassifier(n_estimators=100, min_samples_split=24, random_state=42)
model.fit(X_train, y_train)
display(model)

display(Markdown("##### Accuracy Score"))
y_pred = model.predict(X_test)
display(accuracy_score(y_test, y_pred))

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(xlabel="True Labels", ylabel="Predicted Labels")
plt.show()

display(Markdown("##### 6 Trees"))
models = model.estimators_

fig, ax = plt.subplots(3,2, figsize=(20,30))
tree_style = {"filled":True, "rounded":True, "feature_names":features, "class_names":labels, "max_depth":2, "fontsize":10}
plot_tree(models[0], ax=ax[0,0], **tree_style)
ax[0,0].set(title="Tree #1")
plot_tree(models[19], ax=ax[0,1], **tree_style)
ax[0,1].set(title="Tree #20")
plot_tree(models[39], ax=ax[1,0], **tree_style)
ax[1,0].set(title="Tree #40")
plot_tree(models[59], ax=ax[1,1], **tree_style)
ax[1,1].set(title="Tree #60")
plot_tree(models[79], ax=ax[2,0], **tree_style)
ax[2,0].set(title="Tree #80")
plot_tree(models[99], ax=ax[2,1], **tree_style)
ax[2,1].set(title="Tree #100")
plt.show()