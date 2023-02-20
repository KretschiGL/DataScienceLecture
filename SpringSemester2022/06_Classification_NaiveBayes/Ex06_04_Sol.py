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

display(Markdown("##### Loading data & crossvalidating"))
data = pd.read_csv("./Ex06_04_Data.csv", sep=";")
X_train, X_test, y_train, y_test = train_test_split(data["verified_reviews"], data["feedback"], train_size=.8, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=.1))
display(cross_val_score(model, X_train, y_train, cv=5))

display(Markdown("##### Creating & training model and predicting for test set"))
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=.1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(y_pred[:5])

display(Markdown("##### Confusion Matrix"))
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=False, ax=ax)
ax.set(xlabel="True Label", ylabel="Predicted Label")
plt.show()

display(Markdown("##### Predictions"))
display(Markdown("- \"I love my Alexa\""))
display(model.predict(["I love my Alexa"])[0])

display(Markdown("- \"I hate it!!!!\""))
display(model.predict(["I hate it!!!!"])[0])

display(Markdown("- \"It does not work. Sound quality is bad.\""))
display(model.predict(["It does not work. Sound quality is bad."])[0])

display(Markdown("- \"It's a cool tool. My life got way easier.\""))
display(model.predict(["It's a cool tool. My life got way easier."])[0])

display(Markdown("- \"Just the worst product ever\""))
display(model.predict(["Just the worst product ever"])[0])

display(Markdown("- \"The NSA is probably listening...\""))
display(model.predict(["The NSA is probably listening..."])[0])