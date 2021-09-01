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

display(Markdown("##### Sales per Region"))
regions = ["California", "Portland", "Houston"]
data_sales = data[["Region", "Total Volume", "Year"]][data["Region"].isin(regions)].groupby(["Region", "Year"]).sum().unstack().T
display(data_sales)

display(Markdown("##### Sales per Region in Pies"))
fig, ax = plt.subplots(1,3,figsize=(20,10))
i = 0
for r in regions:
    column = data_sales[r]
    explode = [.1 if row == column.max() else 0 for row in column]
    column.plot.pie(ax=ax[i], explode=explode, autopct="%.2f%%", legend=True)
    i+=1
