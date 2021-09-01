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
from scipy.spatial import Voronoi, voronoi_plot_2d

display(Markdown("##### Loading Data"))
data = pd.read_csv("Ex09_02_Data.csv")
display(data.head(5))

display(Markdown("##### Fitting k-Means(1)"))
model = KMeans(n_clusters=1, random_state=42)
model.fit(data)
display(model)

display(Markdown("#### Sum of Distances for k-Means(1)"))
display(model.inertia_)

display(Markdown("##### k-Means(3)"))
model = KMeans(n_clusters=3, random_state=42)
model.fit(data)
display(model.inertia_)

display(Markdown("##### k_Means(8)"))
model = KMeans(n_clusters=8, random_state=42)
model.fit(data)
display(model.inertia_)

display(Markdown("##### Elbow-Method"))
sqr_dist = []
clusters = range(1,13)
for c in clusters:
    model = KMeans(n_clusters=c, random_state=42).fit(data)
    sqr_dist.append(model.inertia_)
    
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(clusters, sqr_dist, "o-")
plt.show()

display(Markdown("##### k-Means(6)"))
model = KMeans(n_clusters=6, random_state=42)
l_pred = model.fit_predict(data)
centers = model.cluster_centers_
display(Markdown(f"Centers: {centers}"))

display(Markdown("##### Plots"))
fig, ax = plt.subplots(2,2,figsize=(20, 20))

ax[0,0].scatter(data["x"], data["z"], c=l_pred, s=20, cmap="rainbow")
ax[0,0].scatter(centers[:,0], centers[:,2], c="k", s=200, alpha=.5)
ax[0,0].set(xlabel="x", ylabel="z", title="X vs Z")
voronoiXZ = Voronoi(centers[:,[0,2]])
voronoi_plot_2d(voronoiXZ, ax=ax[0,0], show_points=False, show_vertices=False)

ax[0,1].scatter(data["y"], data["z"], c=l_pred, s=20, cmap="rainbow")
ax[0,1].scatter(centers[:,1], centers[:,2], c="k", s=200, alpha=.5)
ax[0,1].set(xlabel="y", ylabel="z", title="Y vs Z")
voronoiYZ = Voronoi(centers[:,[1,2]])
voronoi_plot_2d(voronoiYZ, ax=ax[0,1], show_points=False, show_vertices=False)

ax[1,0].scatter(data["x"], data["y"], c=l_pred, s=20, cmap="rainbow")
ax[1,0].scatter(centers[:,0], centers[:,1], c="k", s=200, alpha=.5)
ax[1,0].set(xlabel="x", ylabel="y", title="X vs Y")
voronoiXY = Voronoi(centers[:,[0,1]])
voronoi_plot_2d(voronoiXY, ax=ax[1,0], show_points=False, show_vertices=False)

ax[1,1].set_axis_off()
plt.show()