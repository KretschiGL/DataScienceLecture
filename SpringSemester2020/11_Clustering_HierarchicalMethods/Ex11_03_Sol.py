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
data = pd.read_csv("./Ex11_03_Data.csv")
display(data.head(5))

display(Markdown("##### Normalizing Data"))
data_min = data.drop("Movie", axis=1)
data_n = normalize(data_min)
data_n = pd.DataFrame(data_n, columns=data_min.columns)
display(data_n.head(5))

display(Markdown("##### Dendrogram"))
graph = linkage(data_n, method="ward")
fig, ax = plt.subplots(figsize=(20,10))
dendrogram(graph, ax=ax)
ax.set(title="Dendrogram")
plt.show()

display(Markdown("##### Dendrogram with 5 Clusters"))
fig, ax = plt.subplots(figsize=(20,10))
dendrogram(graph, ax=ax)
ax.axhline(2.75, color="r", ls="--", lw=2)
ax.set(title="Dendrogram")
plt.show()

display(Markdown("##### Clustering"))
model = AgglomerativeClustering(n_clusters=5)
data_n["Cluster"] = model.fit_predict(data_n)
display(data_n.head(5))

display(Markdown("##### Pairplot"))
sns.pairplot(data_n.drop(["Year"], axis=1), hue="Cluster")
plt.show()