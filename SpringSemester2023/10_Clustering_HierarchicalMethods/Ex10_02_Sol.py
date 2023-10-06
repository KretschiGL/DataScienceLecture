# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex10_02_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Normalizing Data"))
scoreFeatures = ["RottenTomatoes", "AudienceScore"]
openingFeatures = ["TheatersOpenWeek", "OpeningWeekend", "BOAvgOpenWeekend"]
cashFeatures = ["DomesticGross", "ForeignGross", "WorldGross", "Budget", "OpenProfit"]
profitFeatures = ["Profitability"]
data_n = data.drop(["Movie", "Year"], axis=1).copy()
data_n[scoreFeatures] = normalize(data_n[scoreFeatures])
data_n[openingFeatures] = normalize(data_n[openingFeatures])
data_n[cashFeatures] = normalize(data_n[cashFeatures])
data_n[profitFeatures] = MinMaxScaler().fit_transform(data_n[profitFeatures])
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
ax.axhline(6, color="r", ls="--", lw=2)
ax.set(title="Dendrogram")
plt.show()

display(Markdown("##### Clustering"))
model = AgglomerativeClustering(n_clusters=3)
data_n["Cluster"] = model.fit_predict(data_n)
display(data_n.head(5))

display(Markdown("##### Pairplot"))
sns.pairplot(data_n, hue="Cluster", palette="rainbow")
plt.show()
