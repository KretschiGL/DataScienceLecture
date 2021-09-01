# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.cluster import KMeans

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex09_03_Data.csv",sep=";")
display(data.head(5))

display(Markdown("##### Creating Features List"))
features = ["Area_Agri", "Area_Live", "Pop_Density", "Pop"]
display(features)

display(Markdown("##### Area Usage Visualization"))
fig, ax = plt.subplots(figsize=(20,10))
im = ax.scatter(x=data[features[0]], y=data[features[1]], c=data[features[2]], s=data[features[3]]*1000, cmap="rainbow")
fig.colorbar(im, ax=ax)
plt.show()

display(Markdown("##### Evaluating possible Numbers of Clusters"))
sqr_dist = []
clust = range(1,20)
for i in clust:
    model = KMeans(n_clusters=i, random_state=42).fit(data[features])
    sqr_dist.append(model.inertia_)
    
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(clust, sqr_dist, "o-")
ax.set(xlabel="Number of Clusters", ylabel="Sum of Distances in Clusters")
plt.show()

display(Markdown("##### Creating and Using the Model"))
model = KMeans(n_clusters=8, random_state=42)
data["Label"] = model.fit_predict(data[features])
centers = pd.DataFrame(model.cluster_centers_, columns=features)
display(data.head(5))

display(Markdown("##### Visualizing the Clusters"))
fig, ax = plt.subplots(figsize=(16,10))
im = ax.scatter(x=data[features[0]], y=data[features[1]], c=data["Label"], s=data[features[3]]*1000, cmap="rainbow")
ax.scatter(x=centers[features[0]], y=centers[features[1]], c="k", s=200, alpha=.5)
plt.show()

display(Markdown("##### Showing the Data in Context of Switzerland"))
fig, ax = plt.subplots(figsize=(30,20))
rappi_name = "Rapperswil-Jona"
rappi = data[data["City"]==rappi_name]
rappi.plot.scatter(ax=ax, x="E", y="N", c="k", s=200, alpha=.5)

hometown_name="Glarus"
hometown = data[data["City"]==hometown_name]
hometown.plot.scatter(ax=ax, x="E", y="N", c="k", s=200, alpha=.5)

data.plot.scatter(ax=ax, x="E", y="N", c="Label", cmap="rainbow", colorbar=False)

ax.annotate(rappi_name, xy=(rappi["E"], rappi["N"]), xytext=(10,0), textcoords="offset points", size=14)
ax.annotate(hometown_name, xy=(hometown["E"], hometown["N"]), xytext=(20,-40), textcoords="offset points", size=14, arrowprops=dict(arrowstyle="->", ec="#ff0000",lw=4))
plt.show()
