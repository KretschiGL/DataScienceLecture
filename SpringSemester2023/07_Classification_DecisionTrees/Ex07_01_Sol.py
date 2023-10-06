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
data = pd.read_csv("./Ex07_01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Number of Passwords"))
display(len(data))

display(Markdown("##### Defining Labels"))
labels = ["weak", "medium", "strong"]
display(labels)

display(Markdown("##### Defining Features"))
features = data.columns.drop(["strength", "password"]).values
display(features)

display(Markdown("##### Splitting Features from Labels"))
X = data[features]
y = data["strength"]
display(X.head(5))
display(y.head(5))

display(Markdown("##### Creating Train and Test sets"))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.66, random_state=42)
display(f"Train set size: {len(X_train)}")
display(f"Test set size: {len(X_test)}")

display(Markdown("##### Creating, training and using the model"))
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(f"Accuracy: {accuracy_score(y_test, y_pred)}")

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(xlabel="True Strength", ylabel="Predicted Strength")
plt.show()

display(Markdown("##### Tree Plot"))
fig, ax = plt.subplots(figsize=(10, 8))
plot_tree(model, ax=ax, filled=True, rounded=True, feature_names=features, class_names=labels)
plt.show()

import re

def decodePassword(pw):
    data = pd.DataFrame()
    data["password"] = pd.Series(pw)
    data["length"] = pd.Series(len(pw))
    data["lower"] = pd.Series(len(re.findall("[a-z]", pw)))
    data["upper"] = pd.Series(len(re.findall("[A-Z]", pw)))
    data["digits"] = pd.Series(len(re.findall("[0-9]", pw)))
    data["special_chars"] = pd.Series(len(re.findall("[^a-zA-Z0-9]", pw)))
    data = data.set_index("password")
    return data

display(Markdown("##### Predicting"))
s = model.predict(decodePassword("asdf"))
display(Markdown(f"*asdf* is {labels[s[0]]}"))

s = model.predict(decodePassword("Admin1234"))
display(Markdown(f"*Admin1234* is {labels[s[0]]}"))

s = model.predict(decodePassword("WellThisPWisVeryLong"))
display(Markdown(f"*WellThisPWisVeryLong* is {labels[s[0]]}"))

s = model.predict(decodePassword("$bwZKaw.T34o2!"))
display(Markdown(f"*$bwZKaw.T34o2!* is {labels[s[0]]}"))