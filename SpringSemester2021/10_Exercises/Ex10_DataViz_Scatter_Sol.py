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

display(Markdown("##### Preparing the Data"))
grp = data[data["Region"] != "TotalUS"].groupby(["Region","Year"])
data_sales = grp[["Total Bags", "Total Volume"]].sum()
data_sales["AveragePrice"] = grp["AveragePrice"].mean()
display(data_sales)

display(Markdown("##### Plotting the Bags/Volume Compariosn"))
fig, ax = plt.subplots(figsize=(30,10))
data_sales.reset_index(1).plot.scatter(ax=ax, x="Total Bags", y="Total Volume", c="Year", colormap="rainbow", s=data_sales["AveragePrice"]*10)
