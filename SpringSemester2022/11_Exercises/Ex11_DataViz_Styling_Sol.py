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

display(Markdown("##### Total Volume per Region"))
styles = { 
    "West":"--b",
    "Northeast" : "-og",
    "Southeast": "or"
}
fig, ax = plt.subplots(figsize=(20, 5))
for r in styles:
    data[data["Region"] == r].sort_values("Date").plot("Date", "Total Volume", style=styles[r], ax=ax, label=r)
