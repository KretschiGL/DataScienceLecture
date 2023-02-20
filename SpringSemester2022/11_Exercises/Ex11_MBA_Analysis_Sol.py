# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

display(Markdown("##### Simple MBA"))
data = pd.read_csv("./Ex11_MBA_Data.csv", sep=";")
fi_sets = apriori(data, min_support=.05, use_colnames=True)
rules = association_rules(fi_sets, metric="lift", min_threshold=1).sort_values("lift", ascending=False)
display(rules)
