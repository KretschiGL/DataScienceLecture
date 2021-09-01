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

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex12_02_Data.csv")
display(data.head(5))

display(Markdown("##### Plotting Data"))
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter("x", "y", ax=ax, c="label", cmap="rainbow", colorbar=False)
plt.show()

display(Markdown("##### Running DENCLUE"))
display(Markdown("This will take a while..."))
model = DENCLUE()
model.fit(data[["x", "y"]].to_numpy())
display(model)

display(Markdown("##### Number of Clusters"))
display(Markdown(f"\# Clusters: {len(model.clust_info_)}"))

display(Markdown("##### Centroids"))

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

centroids = get_centroids(model)
display(centroids)

display(Markdown("##### Plotting Clusters"))
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter("x", "y", ax=ax, c=model.labels_, cmap="rainbow", colorbar=False)
centroids.plot.scatter("x", "y", ax=ax, c="k", s=200, alpha=.5)
plt.show()

display(Markdown("##### Chaning Density-Limit"))
model.set_minimum_density(0.04)
display(model)

display(Markdown("##### Plotting Clusters again"))
centroids = get_centroids(model, min_density=0.04)
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter("x", "y", ax=ax, c=model.labels_, cmap="rainbow", colorbar=False)
centroids.plot.scatter("x", "y", ax=ax, c="k", s=200, alpha=.5)
plt.show()
