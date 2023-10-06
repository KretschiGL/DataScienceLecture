# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

data = pd.read_csv("./Ex13_DataViz_Data.csv", sep=";")

display(Markdown("##### Total Bags sold per Year"))
fig, ax = plt.subplots(figsize=(10,5))
data[data["Region"] == "TotalUS"][["Total Bags", "Year"]].groupby("Year").sum().plot.bar(ax=ax, rot=0)
