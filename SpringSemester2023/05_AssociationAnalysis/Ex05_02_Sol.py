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

display(Markdown("##### Loading"))
data = pd.read_csv("./Ex05_02_Data.csv", sep=";")
data = data.astype(bool)
display(data.head(5))

display(Markdown("##### Number of Carts"))
display(len(data))

display(Markdown("##### Frequent Item Sets"))
itemset = apriori(data, min_support=.3, use_colnames=True)
display(itemset)

display(Markdown("##### Rules"))
rules = association_rules(itemset, metric="confidence", min_threshold=.5)
display(rules)

display(Markdown("##### Number of Rules"))
display(len(rules))

display(Markdown("##### Rules ordered by their `lift`"))
rules = rules.sort_values("lift", ascending=False)
display(rules)