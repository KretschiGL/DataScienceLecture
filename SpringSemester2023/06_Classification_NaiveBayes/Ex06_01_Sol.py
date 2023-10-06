# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### Defining Categories"))
categories = ['alt.atheism',  'comp.graphics',  'comp.os.ms-windows.misc',  'comp.sys.ibm.pc.hardware',  'comp.sys.mac.hardware',  'comp.windows.x',  'misc.forsale', 
              'rec.autos',  'rec.motorcycles',  'rec.sport.baseball',  'rec.sport.hockey']
display(categories)

display(Markdown("##### Getting `train` and `test` data"))
from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)
display("train = " + train.data[0][:100])
display("test = " + test.data[0][:100])

display(Markdown("##### Creating the model"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
display(model)

display(Markdown("##### Training the model"))
model.fit(train["data"], train["target"])
display(model)

display(Markdown("##### Predicting categories for test data"))
pred_test = model.predict(test["data"])
display(pred_test)

display(Markdown("##### Creating the confusion matrix"))
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test["target"], pred_test)
display(mat)

display(Markdown("##### Plotting the heatmap"))
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False, xticklabels=train.target_names, yticklabels=train.target_names, ax=ax)
ax.set(xlabel="True Category", ylabel="Predicted Category")
plt.show()

display(Markdown("##### Prediciting"))
display(Markdown("- General Motors is a car manufacturer."))
cat = model.predict(["General Motors is a car manufacturer."])
display(train["target_names"][cat[0]])

display(Markdown("- The Boston Red Sox actually wear red socks."))
cat = model.predict(["The Boston Red Sox actually wear red socks."])
display(train["target_names"][cat[0]])

display(Markdown("- Have you tried turning it off and on again? Maybe a reboot helps."))
cat = model.predict(["Have you tried turning it off and on again? Maybe a reboot helps."])
display(train["target_names"][cat[0]])