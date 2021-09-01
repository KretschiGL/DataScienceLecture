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

display(Markdown("##### Volume per PLU over Time"))
fig, ax = plt.subplots(figsize=(30,10))
data[data["Region"] != "TotalUS"].groupby("Date")[["4046", "4225", "4770"]].sum().sort_index().plot.area(ax=ax)
