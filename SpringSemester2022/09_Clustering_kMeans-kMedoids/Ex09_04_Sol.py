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
from mpl_toolkits.mplot3d import Axes3D

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex09_04_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Finding Number of Clusters"))
sqr_dist = []
clusters = range(1,13)
for c in clusters:
    model = KMeans(n_clusters=c, random_state=42).fit(data)
    sqr_dist.append(model.inertia_)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(clusters, sqr_dist, "o-")
plt.show()

display(Markdown("##### k-Means(8)"))
model = KMeans(n_clusters=8, random_state=42)
l_pred = model.fit_predict(data)
display(Markdown(f"Labels: {l_pred[:10]}..."))

display(Markdown("##### Centers"))
centers = model.cluster_centers_
display(centers)

display(Markdown("##### 3D Plot"))
fig, ax = plt.subplots(figsize=(20,15), subplot_kw={"projection":"3d"})
ax.scatter(data["x"], data["y"], data["z"], c=l_pred, s=10, cmap="rainbow", alpha=.5)
ax.scatter(centers[:,0], centers[:,1], centers[:,2], c="k", s=200)
plt.show()