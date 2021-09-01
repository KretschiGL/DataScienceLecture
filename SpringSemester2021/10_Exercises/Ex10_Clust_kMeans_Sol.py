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

display(Markdown("##### Mall Data"))
data = pd.read_csv("./Ex10_Clust_kMeans_Data.csv")
display(data.head(5))

features = ["Age", "Income(k)", "SpendingScore"]

display(Markdown("##### Finding a good Number of Clusters"))
dist = []
clusters = range(1,20)
for c in clusters:
    model = KMeans(n_clusters=c, random_state=42).fit(data[features])
    dist.append(model.inertia_)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(clusters, dist, "o-")
plt.show()

display(Markdown("##### Defining Labels"))
model = KMeans(n_clusters=6, random_state=42)
data["Label"] = model.fit_predict(data[features])
display(data.head(5))

display(Markdown("##### Cluster Centers"))
centers = pd.DataFrame(model.cluster_centers_, columns=features)
display(centers)

display(Markdown("##### Cluster Visualization"))
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter(ax=ax, x="Income(k)", y="SpendingScore", c="Label", cmap="rainbow")
centers.plot.scatter(ax=ax, x="Income(k)", y="SpendingScore", c="k", s=100, alpha=.5)
plt.show()

display(Markdown("##### Age Histograms"))
fig, ax = plt.subplots(2,3, figsize=(20,5), sharex=True, sharey=True)
for i in range(6):
    col = i % 3
    row = i // 3
    data[data["Label"] == i]["Age"].plot.hist(ax=ax[row,col])
plt.show()

display(Markdown("##### Gender Distribution"))
fig, ax = plt.subplots(figsize=(5,5))
data.groupby("Label")[["Male", "Female"]].count().plot.bar(ax=ax, rot=0)
plt.show()
