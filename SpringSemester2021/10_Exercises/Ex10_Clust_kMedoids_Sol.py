# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn_extra.cluster import KMedoids

display(Markdown("##### Credit Card Data"))
data = pd.read_csv("./Ex10_Clust_kMedoids_Data.csv")
display(data.head(5))

features = data.columns.drop("CUST_ID").values

display(Markdown("##### Finding Number of Clusters"))
dist = []
clusters = range(1,20)
for c in clusters:
    model = KMedoids(n_clusters=c, init="k-medoids++", random_state=42).fit(data[features])
    dist.append(model.inertia_)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(clusters, dist, "o-")
plt.show()

display(Markdown("##### Running Cluster Analysis"))
model = KMedoids(n_clusters=9, init="k-medoids++", random_state=42)
data["Label"] = model.fit_predict(data[features])
centers = pd.DataFrame(model.cluster_centers_, columns=features)
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter(ax=ax, x="PURCHASES_FREQUENCY", y="BALANCE", c="Label", s=data["PAYMENTS"]*100, cmap="rainbow")
centers.plot.scatter(ax=ax, x="PURCHASES_FREQUENCY", y="BALANCE", c="k", s=100, alpha=.5)
plt.show()

display(Markdown("##### Mean, StdDev & Median per Cluster"))
summary = data.groupby("Label").agg([np.mean, np.std, np.median])
display(summary)
