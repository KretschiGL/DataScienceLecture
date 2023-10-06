# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex10_PCA_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Scaling Data"))
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
display(data_scaled.head(5))

display(Markdown("##### Variance per Principal Component"))
pca = PCA()
pca.fit(data_scaled)
display(np.around(pca.explained_variance_ratio_, 3))

display(Markdown("##### CumSum Plot of Variance"))
n = range(1, len(data_scaled.columns) + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n, pca.explained_variance_ratio_.cumsum(), "o--")
plt.show()

display(Markdown("##### PCA with 2 Components"))
pca = PCA(n_components=2)
data_t = pca.fit_transform(data_scaled)
data_p = pd.DataFrame(data_t, columns=["PC1", "PC2"])
data_p = pd.concat([data_p, data], axis=1)
display(data_p.head(5))

display(Markdown("##### Principal Components Plot"))
fig, ax = plt.subplots(2,2, figsize=(20, 20))
labels = {"xlabel":"PC1", "ylabel":"PC2"}
fig.suptitle("PCA Plots")
data_p.plot.scatter("PC1", "PC2", ax=ax[0,0], c="Income", cmap="rainbow")
ax[0,0].set(title="Income", **labels)
data_p.plot.scatter("PC1", "PC2", ax=ax[0,1], c="Age", cmap="rainbow")
ax[0,1].set(title="Age", **labels)
data_p.plot.scatter("PC1", "PC2", ax=ax[1,0], c="Education", cmap="rainbow")
ax[1,0].set(title="Education", **labels)
data_p.plot.scatter("PC1", "PC2", ax=ax[1,1], c="Gender", cmap="rainbow")
ax[1,1].set(title="Gender", **labels)
plt.show()
