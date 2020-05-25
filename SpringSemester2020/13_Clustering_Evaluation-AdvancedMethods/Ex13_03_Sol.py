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

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex13_03_Data.csv")
display(data.head(5))

display(Markdown("##### Run DBSCAN"))
model = DBSCAN()
l_pred = model.fit_predict(data)
display(model)

display(Markdown("##### Plotting the Clusters"))
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter("x", "y", ax=ax, c=l_pred, cmap="rainbow", colorbar=False)
plt.show()

display(Markdown("##### Plotting the Clusters with Outliers"))
data_clusters = data[l_pred != -1]
labels_clusters = l_pred[l_pred != -1]
data_outliers = data[l_pred == -1]
fig, ax = plt.subplots(figsize=(10,10))
data_outliers.plot.scatter("x", "y", ax=ax, c="k")
data_clusters.plot.scatter("x", "y", ax=ax, c=labels_clusters, cmap="rainbow", colorbar=False)
plt.show()

display(Markdown("##### Running DBSCAN (eps=.1)"))
model = DBSCAN(eps=.1)
l_pred = model.fit_predict(data)
data_clusters = data[l_pred != -1]
labels_clusters = l_pred[l_pred != -1]
data_outliers = data[l_pred == -1]
fig, ax = plt.subplots(figsize=(10,10))
data_outliers.plot.scatter("x", "y", ax=ax, c="k")
data_clusters.plot.scatter("x", "y", ax=ax, c=labels_clusters, cmap="rainbow", colorbar=False)
plt.show()