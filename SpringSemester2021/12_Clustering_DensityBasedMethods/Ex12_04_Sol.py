# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from denclue import DENCLUE

display(Markdown("##### Loading Airbnb Listings"))
data = pd.read_csv("./Ex12_04_Data.csv")
display(data.head(5))

display(Markdown("##### Zurich"))
fig, ax = plt.subplots(figsize=(15,10))
data.plot.scatter(ax=ax, x="longitude", y="latitude", c="availability_365", cmap="rainbow")
plt.show()

display(Markdown("##### Home/Apartment Subset"))
data_clust = data[(data["room_type"] == "Entire home/apt") & (data["availability_365"] > 300)]
display(Markdown(f"Dataset size: {len(data_clust)}"))
display(data_clust.head(5))

display(Markdown("##### Running DENCLUE"))
display(Markdown("This will take a while..."))
model = DENCLUE(h=.003)
model.fit(data_clust[["longitude", "latitude"]].to_numpy())
display(model)

display(Markdown("##### Setting Minimum Density"))
model.set_minimum_density(300)
display(model)

display(Markdown("##### Getting Centroids"))
def get_centroids(model, min_density=0.0):
    centroids = pd.DataFrame(columns=["x", "y", "density"])
    for i in range(len(model.clust_info_)):
        clust = model.clust_info_[i]
        if(clust["density"] < min_density):
            continue
        centroid = pd.DataFrame([clust["centroid"]], columns=["x", "y"])
        centroid["density"] = clust["density"]
        centroids = centroids.append(centroid, ignore_index=True)
    return centroids

centroids = get_centroids(model, min_density=300)
display(centroids)

display(Markdown("##### Plotting Clusters"))
fig, ax = plt.subplots(figsize=(20,20))
clusters = model.labels_ != -1
outliers = model.labels_ == -1
data_clust[outliers].plot.scatter(ax=ax, x="longitude", y="latitude", c="k", alpha=.2)
data_clust[clusters].plot.scatter(ax=ax, x="longitude", y="latitude", c=model.labels_[clusters], cmap="rainbow")
centroids.plot.scatter(ax=ax, x="x", y="y", c="k", s=200, alpha=.5)
plt.show()
