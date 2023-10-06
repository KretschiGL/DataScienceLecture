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
data = pd.read_csv("./Ex11_03_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Plotting Data"))
fig, ax = plt.subplots(figsize=(20,10))
data.plot.scatter("x", "y", ax=ax, c="label", cmap="rainbow", colorbar=False)
plt.show()

display(Markdown("##### Running DENCLUE"))
display(Markdown("This may take a while..."))
model = DENCLUE(h=.75, min_density=.01)
model.fit(data[["x", "y"]].to_numpy())
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
        centroids = pd.concat([centroids, centroid], ignore_index=True)
    return centroids

centroids = get_centroids(model, min_density=.01)
display(centroids)

display(Markdown("##### Number of Clusters"))
display(Markdown(f"\# Clusters: {len(centroids)}"))

display(Markdown("##### Plotting the Data with Clusters"))
fig, ax = plt.subplots(figsize=(20,10))
data.plot.scatter("x", "y", ax=ax, c=model.labels_, cmap="rainbow", colorbar=False)
centroids.plot.scatter("x", "y", ax=ax, c="k", s=200, alpha=.5)
plt.show()

display(Markdown("##### Trying to find a good model"))
display(Markdown("This is just a suggestion to start from."))
model = DENCLUE(h=.11, eps=.005, min_density=0.1)
display(model)
display(Markdown("Calculating..."))
model.fit(data[["x", "y"]].to_numpy())
centroids = get_centroids(model, min_density=.1)
fig, ax = plt.subplots(figsize=(20,10))
data.plot.scatter("x", "y", ax=ax, c=model.labels_, cmap="rainbow", colorbar=False)
centroids.plot.scatter("x", "y", ax=ax, c="k", s=200, alpha=.5)
plt.show()
