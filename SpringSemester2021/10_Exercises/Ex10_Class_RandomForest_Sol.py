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
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

display(Markdown("##### Raw Data"))
data = pd.read_csv("./Ex10_Class_RandomForest_Data.csv")
display(data.head(5))

display(Markdown("##### Filtered Data"))
col_against = data.columns[data.columns.str.startswith("against_")]
data_tree = data[col_against].copy()
data_tree["type"] = data["type1"]
display(data_tree.head(5))

display(Markdown("##### Parameter Search"))
params = {
    "max_features" : list(range(len(col_against)//2, len(col_against))) + [None],
    "max_depth" : list(range(4, 12)) + [None]
}
submodel = RandomForestClassifier(n_estimators=100, random_state=42)
grid = GridSearchCV(submodel, params, cv=6, n_jobs=4)
grid.fit(data_tree[col_against], data_tree["type"])
display(Markdown(f"Parameters: {grid.best_params_}"))
display(Markdown(f"Score: {grid.best_score_}"))

X_train, X_test, y_train, y_test = train_test_split(data_tree[col_against], data_tree["type"], train_size=.8, random_state=42)
model = grid.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=True, xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set(xlabel="True Types", ylabel="Predicted Types")
plt.show()

display(Markdown("##### Decision Trees"))
sns.reset_orig()
fig, ax = plt.subplots(2,1, figsize=(20, 20))
models = model.estimators_
features = col_against.values
labels = model.classes_
tree_style = {"filled": True, "rounded": True, "feature_names": features, "class_names": labels, "max_depth":3, "fontsize":10}
d = plot_tree(models[21], ax=ax[0], **tree_style)
d = plot_tree(models[42], ax=ax[1], **tree_style)
sns.set()
