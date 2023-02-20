# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

data = pd.read_csv("./Ex11_DataViz_Data.csv", sep=";")

fig, ax = plt.subplots(figsize=(20, 20))
idx = data.groupby("Region")["Total Volume"].sum().sort_values().index.drop("TotalUS")
data[data["Region"] != "TotalUS"][["Region", "Total Volume", "Year"]].groupby(["Region", "Year"]).sum().unstack().loc[idx].plot.barh(ax=ax, stacked=True)
