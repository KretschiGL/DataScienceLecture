# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

display(Markdown("##### Loading Data"))
data = pd.read_csv("Ex10_01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Normalize Data"))
nonfeatures = ["Category", "Item", "Serving Size", "Calories", "Calories from Fat"]
features = [c for c in data.columns.values if c not in nonfeatures]
data_n = pd.DataFrame(normalize(data[features]), columns=features)
display(data_n.head(5))

display(Markdown("##### Dendrogram"))
graph = linkage(data_n, method="ward")
fig, ax = plt.subplots(figsize=(20,10))
dendrogram(graph, ax=ax)
ax.set(title="McDonald's Menu Dendrogram")
plt.show()

display(Markdown("##### Dendrogram with Cut"))
fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(graph, ax=ax)
ax.axhline(3, color="r", ls="--", lw=2)
ax.set(title="McDonald's Menu Dendrogram")
plt.show()

display(Markdown("##### Data with Cluster"))
model = AgglomerativeClustering(n_clusters=3)
data["Cluster"] = model.fit_predict(data_n)
display(data.head(5))

display(Markdown("##### Category-Cluster Bar Chart"))
fig, ax = plt.subplots(figsize=(20,10))
data.groupby(["Category", "Cluster"])["Cluster"].count().unstack().plot.bar(ax=ax, rot=0)
plt.show()

display(Markdown("##### Mean & Median Values of the Data"))
mean = data.groupby("Cluster")[features].agg([np.mean, np.median])
display(mean)

display(Markdown("##### Feature Comparison"))
sns.pairplot(data, hue="Cluster", palette="rainbow", vars=features)
plt.show()

display(Markdown("##### Closer Look"))
fig, ax = plt.subplots(1,2,figsize=(20,10))
data.plot.scatter(ax=ax[0], x="Sugars", y="Saturated Fat", c="Cluster", s="Calories", cmap="rainbow", colorbar=False)
data.plot.scatter(ax=ax[1], x="Protein", y="Sodium", c="Cluster", s="Calories", cmap="rainbow", colorbar=False)
plt.show()
