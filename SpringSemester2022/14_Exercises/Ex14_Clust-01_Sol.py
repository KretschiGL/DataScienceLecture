# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.cluster import DBSCAN
from sklearn.base import clone

display(Markdown("###### Loading Wi-Fi Data"))
data = pd.read_csv("./Ex14_Clust-01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("###### NYC Plot"))
fig, ax = plt.subplots(figsize=(20,20))
data.plot.scatter("Longitude", "Latitude", ax=ax, c="b")
fig.suptitle("Wi-Fi Hotspots in NYC")
plt.show()

display(Markdown("###### Clustering"))

def clustering(data, model, metric, ax):
    m = clone(model)
    m.set_params(metric=metric)
    l_pred = m.fit_predict(data)
    n_cluster = len(np.unique(l_pred))
    data_cluster = data[l_pred != -1]
    label_cluster = l_pred[l_pred != -1]
    data_outlier = data[l_pred == -1]
    data_outlier.plot.scatter("Longitude", "Latitude", ax=ax, c="k", alpha=.5)
    data_cluster.plot.scatter("Longitude", "Latitude", ax=ax, c=label_cluster, cmap="rainbow", colorbar=False)
    ax.set(title=f"Found {n_cluster} clusters with distance metric {metric}")

model = DBSCAN(eps=.005)
data_coord = data[["Longitude", "Latitude"]]
fig, ax = plt.subplots(1,2,figsize=(20,10))
clustering(data_coord, model, "euclidean", ax[0])
clustering(data_coord, model, "manhattan", ax[1])
fig.suptitle("Wi-Fi Clusters in NYC")
