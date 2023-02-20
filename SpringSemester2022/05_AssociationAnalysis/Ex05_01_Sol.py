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
data = pd.read_csv("./Ex05_01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Preprocessing"))
data = data.replace("X", 1).fillna(0).astype(int)
display(data.head(5))

display(Markdown("##### Item sets"))
itemset = apriori(data, min_support=.3, use_colnames=True)
display(itemset)

display(Markdown("##### Rules"))
rules = association_rules(itemset, metric="confidence", min_threshold=.85)
display(rules)