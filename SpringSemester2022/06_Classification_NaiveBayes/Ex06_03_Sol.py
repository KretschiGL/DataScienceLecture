# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown
# Init Solution completed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

display(Markdown("##### Loading data"))
data = pd.read_csv("./Ex06_03_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Creating train and test sets"))
X_train, X_test, y_train, y_test = train_test_split(data["verified_reviews"], data["feedback"], train_size=.8, random_state=42)
display(Markdown(f"Train set size: {len(X_train)/len(data):.2f}%"))

display(Markdown("##### Crossvalidation"))
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
display(cross_val_score(model, X_train, y_train, cv=5))

display(Markdown("##### Creating model, training and prediction"))
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(model)

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=False, ax=ax)
ax.set(xlabel="True Label", ylabel="Predicted Label")
plt.show()

display(Markdown("##### Class imbalance"))
data["feedback"].value_counts().plot.bar()
plt.show()

display(Markdown("##### Upsampling"))
data_pos = data[data["feedback"] == 1]
data_neg = resample(data[data["feedback"] == 0], replace=True, n_samples=int(.66*len(data_pos)), random_state=42)
data_new = pd.concat([data_pos, data_neg])
display(Markdown(f"Dataset size before upsampling: {len(data)}, after upsampling: {len(data_new)}"))

display(Markdown("##### Class imbalance gone"))
data_new["feedback"].value_counts().plot.bar()
plt.show()