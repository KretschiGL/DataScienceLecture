# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix 

display(Markdown("###### Loading Reviews"))
data = pd.read_csv("./Ex14_Class-01_Data-Train.csv", sep=";")
display(data.head(5))

display(Markdown("###### Training the Model"))
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["Class"], train_size=.8, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(Markdown(f"Accuracy score: {accuracy_score(y_test, y_pred)}"))
labels = data["Class"].unique()
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(matrix.T, ax=ax, square=True, annot=True, fmt="d", cbar=False, xticklabels=labels, yticklabels=labels)
ax.set(title="Ford vs Not Ford Confusion Matrix",xlabel="True Labels", ylabel="Predicted Labels")
plt.show()
cvscores = cross_val_score(model, data["Text"], data["Class"], cv=5)
display(Markdown(f"Cross validation scores: {cvscores} => Avg: {cvscores.mean()}"))

display(Markdown("###### Retrain the Model"))
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data["Text"], data["Class"])
display(model)

display(Markdown("###### New Reviews with predicted Classes"))
data_holdout = pd.read_csv("./Ex14_Class-01_Data-Holdout.csv", sep=";")
data_holdout["Class"] = model.predict(data_holdout["Text"])
display(data_holdout.head(5))
