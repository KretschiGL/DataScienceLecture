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
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.base import clone
from matplotlib import cm

def silhouettes(data, model, n_clusters):
    m = clone(model)                                 # Create copy of the model
    m.set_params(n_clusters=n_clusters)              # Setting the numbers of clusters
    l_pred = m.fit_predict(data)
    coef = silhouette_score(data, l_pred)            # Get the silhouette coefficient
    print(f"Silhouette coefficient for {n_clusters}: {np.round(coef, 6)}")
    values = silhouette_samples(data, l_pred)        # Get the silhouette values
    colors = cm.get_cmap("rainbow", n_clusters)      # Getting colors
    fig, ax = plt.subplots(figsize=(10,10))
    fig.suptitle(f"Silhouette Analysis with {n_clusters} Clusters")
    margin = 10
    y_bottom = margin
    # Plotting the silhouette values
    for c in range(n_clusters):
        values_c = values[l_pred == c]               # Just getting values of cluster c
        values_c = np.sort(values_c)                 # Sorting the values
        size_c = len(values_c)                       # Getting the number of values
        y_top = y_bottom + size_c                    # Calculating the value on the y-axis to draw to
        color = colors(c/n_clusters)                 # Selecting the color of the cluster
        # Plotting the values
        (ax.fill_betweenx(                        # Fill an area
            np.arange(y_bottom, y_top),              # Range on y-axis to fill
            0,                                       # Start value on x-axis
            values_c,                                # End value on x-axis
            fc=color, ec=color, alpha=.7))           # Some styling
        # Plotting the cluster number to the left
        ax.text(-.05, y_bottom + .5 * size_c, str(c))
        y_bottom = y_top + margin                    # Calculating the next start value on the y-axis
    ax.axvline(coef, c="k", ls="--")              # Plotting a vertical line for the silhouette coefficient
    ax.set(title="Silhouette Values", xlabel="Silhouette Values", ylabel="Clusters", yticks=[], xlim=(-.1,1))

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex13_01_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Dataset without Labels"))
data_s = data.drop("label", axis="columns")
display(data_s.head(5))

display(Markdown("##### Silhouette Coefficient (2 Clusters)"))
model = KMeans(n_clusters=2, random_state=42)
l_pred = model.fit_predict(data_s)
coef = silhouette_score(data_s, l_pred)
display(Markdown(f"Silhouette coefficient with 2 clusters = {coef}"))

display(Markdown("##### Silhouette Coefficient (5 Clusters)"))
model = KMeans(n_clusters=5, random_state=42)
l_pred = model.fit_predict(data_s)
coef = silhouette_score(data_s, l_pred)
display(Markdown(f"Silhouette coefficient with 5 clusters = {coef}"))

display(Markdown("##### Avg. Silhouette Value (5 Clusters)"))
values = silhouette_samples(data_s, l_pred)
for c in range(5):
    v = values[l_pred==c]
    display(Markdown(f"Silhouette coefficient of cluster {c}: {np.mean(v)}"))

display(Markdown("##### Silhouette Coefficients for 2 - 10 Clusters"))
for c in range(2,11):
    model = KMeans(n_clusters=c, random_state=42)
    l_pred = model.fit_predict(data_s)
    display(Markdown(f"Silhouette coefficient with {c} clusters: {silhouette_score(data_s, l_pred)}"))

display(Markdown("##### Silhouette Analysis for 2 - 10 Clusters"))
model = KMeans(random_state=42)
clusters = range(2,11)
for c in clusters:
    silhouettes(data_s, model, c)
plt.show()

display(Markdown("##### Showing the expected 8 Clusters"))
model = KMeans(n_clusters=8, random_state=42)
l_pred = model.fit_predict(data_s)
fig, ax = plt.subplots(figsize=(10,10))
data_s.plot.scatter("x", "y", ax=ax, c=l_pred, cmap="rainbow", colorbar=False)
plt.show()
