# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown
# Init Solution completed


display(Markdown("##### Loading the Data"))
airbnb = pd.read_csv("./Ex04_04_Data.csv")
display(airbnb.head(5))


display(Markdown("##### Preprocessing - adding new columns"))
airbnb["min_price_per_stay"] = airbnb["price"] * airbnb["minimum_nights"]
airbnb["revenue_365"] = airbnb["price"] * airbnb["availability_365"]
display(airbnb.head(1))


display(Markdown("##### 10 most expensive offerings"))
top10 = airbnb[airbnb["availability_365"] > 0]["min_price_per_stay"].sort_values(ascending=False).head(10)
top10 = airbnb.loc[top10.index]
display(top10)


fig, ax = plt.subplots()
fig.suptitle("Where are the 10 most expensive offerings in NYC?")
top10["neighbourhood_group"].value_counts().plot.bar(ax=ax, rot=0)


fig, ax = plt.subplots()
fig.suptitle("Airbnb Distribution in NYC")
airbnb.groupby("neighbourhood_group")["host_id"].count().rename("Distribution").plot.pie(ax=ax, autopct="%1.2f%%")


fig, ax = plt.subplots()
fig.suptitle("Prices per Neighborhood")
airbnb.groupby("neighbourhood_group")["price"].aggregate([np.mean, np.median]).plot.bar(ax=ax, rot=0)
ax.set(xlabel="Neighborhoods", ylabel="Price")


fig,ax = plt.subplots()
fig.suptitle("Distribution of Room Types")
airbnb["room_type"].value_counts().rename("Room Types").plot.pie(ax=ax, autopct="%1.2f%%")


prices = airbnb.groupby("room_type")["price"].aggregate([np.mean, np.median])
fig, ax = plt.subplots()
fig.suptitle("Prices per Room Type")
prices.plot.bar(ax=ax, rot=0)
ax.set(xlabel="Room Types", ylabel="Price")


hosts = airbnb["host_id"].value_counts()
fig, ax = plt.subplots()
fig.suptitle("Who has how many listings?")
pd.cut(hosts, bins=[0,1,2,3,4,hosts.max()], labels=["1","2","3","4","5+"]).value_counts(sort=False).plot.bar(ax=ax, rot=0)
ax.set(xlabel="Listings per Host", ylabel="Hosts")


display(Markdown("##### Mean number of listings per host"))
display(hosts.mean())


display(Markdown("##### Availability"))
airbnb["availability"] = pd.cut(airbnb["availability_365"], bins=[-1,59,365], labels=["low", "high"])
display(airbnb.head(5))


fig, ax = plt.subplots()
fig.suptitle("Availability of Listings")
airbnb["availability"].value_counts().sort_values().plot.pie(ax=ax, autopct="%1.2f%%", explode=[.1, 0], shadow=True)


# Challenge
revenue = airbnb.groupby("host_id")["revenue_365"].sum().sort_values()
revenue = revenue[revenue > 0].value_counts().reset_index().rename(columns={"index":"revenue", "revenue_365":"hosts"}).sort_values("revenue")
fig, ax = plt.subplots(figsize=(20,5))
fig.suptitle("How many can expecte how nuch revenue?")
revenue[revenue < 100000].plot.line("revenue", "hosts", ax=ax, label="Hosts")
ax.set(xlabel="Revenue", ylabel="Number of Hosts")


# Challenge
areaNames = airbnb["neighbourhood_group"].unique()
areaDict = dict(zip(areaNames, range(len(areaNames))))
display(areaDict)

available = airbnb[(airbnb["availability_365"] > 0) & (airbnb["price"] > 1000)]
colors = available["neighbourhood_group"].replace(areaDict)
sizes = available["price"]/5

fig, ax = plt.subplots(figsize=(20,5))
fig.suptitle("Where are the expensive offerings?")
available.plot.scatter("availability_365", "minimum_nights", c=colors, s=sizes, cmap="rainbow", alpha=.5, ax=ax, colorbar=False)
ax.set(ylim=(-50,400), xlabel="Availability per Year", ylabel="Min Nights per Stay")

colorbar = fig.colorbar(ax=ax, mappable=ax.collections[0])
colorbar.set_ticks(list(areaDict.values()))
colorbar.set_ticklabels(list(areaDict.keys()))