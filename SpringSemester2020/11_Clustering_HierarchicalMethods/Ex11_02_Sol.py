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
data = pd.read_csv("Ex11_02_Data.csv")
display(data.head(5))

display(Markdown("##### Minimize Data"))
data_min = data.drop(["Category", "Item", "Serving Size"], axis=1)
display(data_min.head(5))

display(Markdown("##### Normalize Data"))
data_n = normalize(data_min)
data_n = pd.DataFrame(data_n, columns=data_min.columns)
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
ax.axhline(4, color="r", ls="--", lw=2)
ax.set(title="McDonald's Menu Dendrogram")
plt.show()

display(Markdown("##### Data with Cluster"))
model = AgglomerativeClustering(n_clusters=2)
l_pred = model.fit_predict(data_n)
data["Cluster"] = l_pred
display(data.head(5))

display(Markdown("##### Category-Cluster Bar Chart"))
fig, ax = plt.subplots(figsize=(20,10))
data.groupby(["Category", "Cluster"])["Cluster"].count().unstack().plot.bar(ax=ax, rot=0)
plt.show()