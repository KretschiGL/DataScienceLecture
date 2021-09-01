# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex07_04_Data.csv")
display(data.head(5))

display(Markdown("##### Labels & Features"))
labels = ["No", "Yes"]
features = data.columns.drop(["is_canceled"])
display(labels)
display(features)

display(Markdown("##### Train & Test sets"))
X = data[features]
y = data["is_canceled"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, random_state=42)
display(f"Train set size: {len(X_train)}")
display(f"Test set size: {len(X_test)}")

display(Markdown("##### Creating, training and using the model"))
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(accuracy_score(y_test, y_pred))

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(xlabel="True Labels", ylabel="Predicted Labels")
plt.show()

display(Markdown("##### Tree Plot"))
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, ax=ax, filled=True, rounded=True, feature_names=features, class_names=labels, max_depth=4, fontsize=10)