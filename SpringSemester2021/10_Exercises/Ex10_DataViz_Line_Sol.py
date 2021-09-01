# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

data = pd.read_csv("./Ex10_DataViz_Data.csv")

display(Markdown("##### Avocado Pricing"))
import matplotlib.dates as mdates 
fig, ax = plt.subplots(figsize=(20,5))
for r in ["Boston", "Chicago", "NewYork", "LosAngeles", "SanFrancisco"]:
    data[data["Region"] == r].sort_values("Date").plot("Date", "AveragePrice", label=r, ax=ax)
