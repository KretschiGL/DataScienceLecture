# Your code does not need to contain the display()-wrapper
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

rng = np.random.RandomState(42)
x = rng.rand(50)*100-50
y = rng.rand(50)*100-50
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set(title="Just Points")

fig, ax = plt.subplots()
ax.scatter(x, y, marker="^")
ax.set(title="As Triangles")

sizes = ((x + 50) * (y + 50)/100) + 10
print(f"Check: min = {np.min(sizes)}")
print(f"Check: max = {np.max(sizes)}")
fig, ax = plt.subplots()
ax.scatter(x, y, marker="^", s=sizes)
ax.set(title="With different Sizes")

fig, ax = plt.subplots()
ax.scatter(x, y, marker="^", c=x, cmap="rainbow")
fig.colorbar(ax=ax, mappable=ax.collections[0])
ax.set(title="Colored")

lowerLeft = (x < 0) & (y < 0)
upperRight = (x >= 0) & (y >= 0)
upperLeft = (x < 0) & (y >= 0)
lowerRight = (x >= 0) & (y < 0)
fig, ax = plt.subplots()
ax.scatter(x[lowerLeft], y[lowerLeft], marker="s", color="b", s=150, alpha=.5)
ax.scatter(x[upperRight], y[upperRight], marker="d", color="r", s=150, alpha=.75)
ax.scatter(x[upperLeft], y[upperLeft], marker="o", color="g", s=150, alpha=.25)
ax.scatter(x[lowerRight], y[lowerRight], marker="x", color="k", s=150)
ax.set(title="Different Types of Points")