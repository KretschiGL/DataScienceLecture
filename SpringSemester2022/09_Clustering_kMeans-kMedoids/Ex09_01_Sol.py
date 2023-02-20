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
from sklearn_extra.cluster import KMedoids
from scipy.spatial import Voronoi, voronoi_plot_2d

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex09_01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Number of Points and Clusters"))
display(Markdown(f"Points: {len(data)}"))
display(Markdown(f"Clusters: {len(data['l'].unique())}"))

display(Markdown("##### Creating & training k-Means"))
model = KMeans(n_clusters=8, random_state=42)
model.fit(data[["x", "y"]])
display(model)

display(Markdown("##### k-Means Cluster Centers"))
centers = model.cluster_centers_
display(centers)

display(Markdown("##### Prediction Clusters"))
l_pred = model.predict(data[["x", "y"]])
display(l_pred[:5])

display(Markdown("##### k-Means Plot"))
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(data["x"], data["y"], c=l_pred, s=50, cmap="rainbow")
ax.scatter(centers[:,0], centers[:,1], c="k", s=200, alpha=.5)
voronoi = Voronoi(centers)
f = voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)
plt.show()

display(Markdown("##### k-Medoids Plot"))
model = KMedoids(n_clusters=8, init="k-medoids++", random_state=42)
l_pred = model.fit_predict(data[["x", "y"]])
centers = model.cluster_centers_

fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(data["x"], data["y"], c=l_pred, s=50, cmap="rainbow")
ax.scatter(centers[:,0], centers[:,1], c="k", s=200, alpha=.5)
voronoi = Voronoi(centers)
f = voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)
plt.show()