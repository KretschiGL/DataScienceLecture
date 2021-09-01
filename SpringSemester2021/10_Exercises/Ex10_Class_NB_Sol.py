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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

display(Markdown("##### Raw Data"))
data = pd.read_csv("./Ex10_Class_NB_Data.csv")
data = data[["text", "tag"]]
display(data.head(5))
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
cv = cross_val_score(model, data["text"], data["tag"], cv=5)
display(Markdown("##### Cross Validation"))
display(Markdown(f"Scores: {cv} => {cv.mean()}"))
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["tag"], train_size=.8, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=True, xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set(xlabel="True Category", ylabel="Predicted Category")
display(Markdown("##### Confusion Matrix"))
plt.show()

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data["text"], data["tag"])
reviews = ["I really liked this movie. The cast was amazing.", "The worst, never watching a moving from this directory again", "Had fun. Was good.", "Way to long. The story could have been told in half of the time."]
pred = model.predict(reviews)
display(Markdown("##### Test"))
test = pd.DataFrame({"Reviews":reviews, "Pred": pred})
display(test)
