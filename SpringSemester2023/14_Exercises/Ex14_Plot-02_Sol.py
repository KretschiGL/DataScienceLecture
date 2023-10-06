# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("###### Loading Prices"))
prices = pd.read_csv("./Ex14_Plot-02_Data-TechnologyIndex.csv", sep=";")
display(prices.head(5))

display(Markdown("###### Preprocessing Prices"))
prices[prices.columns.drop("Country")] = prices[prices.columns.drop("Country")].apply(lambda c: c.str.replace("$","",regex=False).str.replace(",","",regex=False).str.strip()).astype(float)
display(prices.head(5))
display(prices.info())

display(Markdown("###### Adding Continent"))
countries = pd.read_csv("./Ex14_Plot-02_Data-Countries.csv", sep=";")
data = pd.merge(prices, countries[["Country_Name", "Continent_Name"]], how="left", left_on="Country", right_on="Country_Name").drop("Country_Name", axis="columns")
data = data.rename(columns={"Continent_Name" : "Continent"})
display(data.head(5))

display(Markdown("###### Adding Population"))
population = pd.read_csv("./Ex14_Plot-02_Data-Population.csv", sep=";")
data = pd.merge(data, population[["Location", "PopTotal"]], how="left", left_on="Country", right_on="Location").drop("Location", axis="columns")
display(data.head(5))

display(Markdown("###### Top10 iPhone Prices"))
top10 = data.loc[data["iPhone"].sort_values(ascending=False).index][:10][::-1]
display(top10)

display(Markdown("###### Macbook vs Windows"))
macwin = data[["Country","MacBook", "Windows Powered"]].copy()
macwin["MacBook > Win"] = macwin["MacBook"]/macwin["Windows Powered"] - 1
macwin["bin"] = pd.cut(macwin["MacBook > Win"], [-np.inf, 0, .25, .5, 1, 2, np.inf])
macwin_parts = pd.DataFrame(macwin["bin"].value_counts(sort=False))
macwin_parts = macwin_parts.rename(columns={"bin": "Count"})
minmax = [macwin_parts["Count"].max(), macwin_parts["Count"].min()]
macwin_parts["Highlight"] = macwin_parts["Count"].apply(lambda v: .25 if v in minmax else 0)
macwin_parts["Label"] = ["<= 0%", "0% - 25%", "25% - 50%", "50% - 100%", "100% - 200%", " > 200%"]
display(macwin_parts)

display(Markdown("###### XBox one vs PS4"))
xboxps4 = data[["Xbox one", "PS4", "Country", "Continent", "PopTotal"]].copy()
continents = xboxps4["Continent"].unique()
continents = dict(zip(continents,range(len(continents))))
xboxps4["ContinentId"] = xboxps4["Continent"].replace(continents)
display(xboxps4)

display(Markdown("###### Smart TVs, Headphones, 2TB HDDs"))
devices = pd.DataFrame(data[data["Country"] != "Venezuela"][["40 inch smart TV", "Brand headphone", "hard drive 2TB"]])
display(devices)

display(Markdown("###### Visualization"))
fig, ax = plt.subplots(2,2,figsize=(20,20))
fig.suptitle("Technology Index")

ax[0,0].set(title="Top 10 iPhone vs Android", xlim=(0,8000))
top10[["Android", "iPhone"]].plot.barh(ax=ax[0,0])
ax[0,0].set_yticklabels(top10["Country"])

ax[0,1].set(title="Mac vs Windows Price Differences")
macwin_parts["Count"].plot.pie(ax=ax[0,1], explode=macwin_parts["Highlight"], autopct="%1.2f%%", shadow=True, labels=macwin_parts["Label"])

ax[1,0].set(title="XBox one vs PS4", xlim=(0,800), ylim=(0,1000))
xboxps4.plot.scatter("Xbox one", "PS4", ax=ax[1,0], s=xboxps4["PopTotal"]/1000, c="ContinentId", colormap="rainbow", colorbar=False, alpha=.8)
ax[1,0].set(xlabel="Avg Price XBox one in $", ylabel="Avg Price PS4 in $", )
colorbar = fig.colorbar(ax=ax[1,0], mappable=ax[1,0].collections[0])
colorbar.set_ticks(list(continents.values()))
colorbar.set_ticklabels(list(continents.keys()))
swiss = xboxps4[xboxps4["Country"] == "Switzerland"]
ax[1,0].annotate("Switzerland", xy=(swiss["Xbox one"], swiss["PS4"]), xytext=(-100, 50), textcoords="offset points", size=16,
                arrowprops=dict(arrowstyle="fancy", ec="r", fc="r", connectionstyle="angle3,angleA=0,angleB=-120"))

ax[1,1].set(title="SmartTV vs Headphones vs 2TB HDD Prices")
devices.plot.hist(bins=25, ax=ax[1,1], alpha=.5)
plt.show()
