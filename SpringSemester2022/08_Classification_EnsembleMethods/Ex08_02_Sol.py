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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex08_02_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Class Imbalance"))
data["campaign_success"].value_counts().plot.bar()
plt.show()

display(Markdown("##### Fixing Class Imbalance"))
camp_success = data[data["campaign_success"] == 1].sample(n=35000, replace=True, random_state=42)
camp_failed = data[data["campaign_success"] == 0]
data_camp = pd.concat([camp_success, camp_failed])
data_camp["campaign_success"].value_counts().plot.bar()
plt.show()

display(Markdown("##### Labels & Features"))
labels = ["Failed", "Success"]
features = data_camp.columns.drop("campaign_success").values
display(Markdown(f"Labels: {labels}"))
display(Markdown(f"Features: {features}"))

display(Markdown("Train & Test sets"))
X = data_camp[features]
y = data_camp["campaign_success"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
display(Markdown(f"Train set size: {len(X_train)}"))
display(Markdown(f"Test set size: {len(X_test)}"))

display(Markdown("##### Training and testing the Model"))
model = RandomForestClassifier(n_estimators=100, max_samples=.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(accuracy_score(y_test, y_pred))

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(xlabel="True Labels", ylabel="Predicted Labels")
plt.show()

display(Markdown("##### Training new Model with all Data"))
model = RandomForestClassifier(n_estimators=100, max_samples=.8, random_state=42)
model.fit(X, y)
display(model)

display(Markdown("##### Loading new Data"))
data_use = pd.read_csv("./Ex08_02_Data_Use.csv", sep=";")
display(data_use.head(5))

display(Markdown("##### Predicting the Probabilities"))
proba = pd.DataFrame(model.predict_proba(data_use), columns=["Camp_Fail", "Camp_Success"])
data_proba = pd.concat([proba[["Camp_Success", "Camp_Fail"]], data_use], axis="columns")
display(data_proba.head(5))

display(Markdown("##### Reordering the data points by their expectd success"))
data_proba = data_proba.sort_values("Camp_Success", ascending=False)
display(data_proba.head(10))