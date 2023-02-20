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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

display(Markdown("##### Loading data"))
data = pd.read_csv("./Ex06_02_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Creating train and test sets"))
X_train, X_test, y_train, y_test = train_test_split(data["Message"], data["Label"], train_size=.7, random_state=42)
display(Markdown(f"Train set size: {len(X_train)/len(data):.2f}"))

display(Markdown("##### Creating, training and predicting"))
model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=.1))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display(y_pred[:5])

display(Markdown("##### Plotting Confusion Matrix"))
labels = ["spam","ham"]
matrix = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=False,
           xticklabels=labels, yticklabels=labels, ax=ax)
ax.set(xlabel="True Labels", ylabel="Predicted Labels")
plt.show()

display(Markdown("##### Predicting"))
display(Markdown("- \"Whazaaaap!\""))
display(Markdown(model.predict(["Whazaaaap!"])[0]))

display(Markdown("- \"Congratulations, you've won the lottery.\""))
display(Markdown(model.predict(["Congratulations, you've won the lottery."])[0]))

display(Markdown("- \"Sorry, I'll be late.\""))
display(Markdown(model.predict(["Sorry, I'll be late."])[0]))

display(Markdown("- \"I'm a nigerian prince who needs to transfer some gold. You can have $1'000'000 if you work with me.\""))
display(Markdown(model.predict(["I'm a nigerian prince who needs to transfer some gold. You can have $1'000'000 if you work with me."])[0]))