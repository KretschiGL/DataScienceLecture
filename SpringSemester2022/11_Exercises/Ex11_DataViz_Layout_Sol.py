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

regions = ["Chicago", "Detroit", "Philadelphia", "SanDiego", "Seattle"]
data_viz = data[data["Region"].isin(regions)].copy()

fig, ax = plt.subplots(2,3, figsize=(30,20))

# Top Left
data_viz[data_viz["Year"] == 2017].groupby("Region")[["Small Bags", "Large Bags", "XLarge Bags"]].sum().plot.bar(ax=ax[0,0], rot=0)

# Top Center
data_vol2016 = data_viz[data_viz["Year"] == 2016].groupby("Region")["Total Volume"].sum()
explode = [.1 if row == data_vol2016.max() else 0 for row in data_vol2016]
data_vol2016.plot.pie(ax=ax[0,1], explode=explode, autopct="%.2f%%")

# Top Right
data_viz_sorted = data_viz.sort_values("Date")
for r in regions:
    data_viz_sorted[data_viz_sorted["Region"] == r].plot(ax=ax[0,2], x="Date", y="AveragePrice", label=r)

# Bottom Left
data_plu = data_viz.groupby(["Region", "Year"])[["4046", "4225", "4770"]].sum().unstack()
data_plu["4046"].plot.barh(ax=ax[1,0], stacked=True, width=.1, position=0)
data_plu["4225"].plot.barh(ax=ax[1,0], stacked=True, width=.1, position=1, legend=False)
data_plu["4770"].plot.barh(ax=ax[1,0], stacked=True, width=.1, position=2, legend=False)

# Bottom Center
data_viz.groupby(["Date", "Region"])["Total Volume"].sum().unstack().plot.area(ax=ax[1,1])

# Bottom Right
data_viz["Month"] = pd.to_datetime(data_viz["Date"]).dt.month
regionids = dict(zip(regions,range(len(regions))))
data_month2015 = data_viz[data_viz["Year"] == 2015].groupby(["Month","Region"])[["4046", "4225", "Total Volume"]].sum().reset_index(1)
data_month2015["RegionId"] = data_month2015["Region"].replace(regionids)
data_month2015.plot.scatter(ax=ax[1,2], x="4046", y="4225", c="RegionId", cmap="rainbow", colorbar=False, s=data_month2015["Total Volume"]/10000)
colorbar = fig.colorbar(ax=ax[1,2], mappable=ax[1,2].collections[0])
colorbar.set_ticks(list(regionids.values()))
colorbar.set_ticklabels(list(regionids.keys()))
