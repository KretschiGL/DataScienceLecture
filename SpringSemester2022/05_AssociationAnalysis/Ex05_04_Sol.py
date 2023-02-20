# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Init Solution completed

display(Markdown("##### Loading & Preprocessing"))
data = pd.read_csv("./Ex05_04_Data.csv")
data = data.set_index(["Country", "InvoiceNo"])
data = data.fillna(0).astype(bool).astype(int)
display(data.head(5))

display(Markdown("##### Switzerland"))
display(Markdown("###### Selecting Invoices"))
swiss = data.loc["Switzerland"]
display(swiss.head(5))

display(Markdown("###### Number of Invoices"))
display(len(swiss))

display(Markdown("###### FI sets and Rules"))
itemsets = apriori(swiss, min_support=.1, use_colnames=True)
display(itemsets)
rules = association_rules(itemsets, metric="confidence", min_threshold=.7)
rules = rules.sort_values("lift", ascending=False)
display(rules.head(5))

display(Markdown("###### Number of Rules"))
display(len(rules))


display(Markdown("##### United Kingdom"))
uk = data.loc["United Kingdom"]

display(Markdown("###### Number of Invoices"))
display(len(uk))

display(Markdown("###### Itemsets with min_support >= .1"))
itemsets = apriori(uk, min_support=.1, use_colnames=True)
display(itemsets)

display(Markdown("###### Rules with min_support >= .03"))
itemsets = apriori(uk, min_support=.03, use_colnames=True)
rules = association_rules(itemsets, metric="confidence", min_threshold=.7)
rules = rules.sort_values("lift", ascending=False)
display(rules)

display(Markdown("###### Are we surprised?"))
display(Markdown("**Nope**"))