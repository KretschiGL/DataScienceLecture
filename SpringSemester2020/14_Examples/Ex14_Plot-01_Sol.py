# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("###### Sin(x) vs Cos(x)"))
x = np.linspace(-6,6, 1000)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, np.sin(x), ":g", label="sin(x)")
ax.plot(x, np.cos(x), "--r", label="cos(x)")
ax.legend()
ax.set(title="sin(x) vs cos(x)", xlabel="x", ylabel="y")
plt.show()

display(Markdown("###### Normal Distribution (Histogram)"))
rng = np.random.RandomState(42)
x = rng.randn(1000000)
fig, ax = plt.subplots(figsize=(20,5))
ax.hist(x, bins=1000, ec="b")
ax.set(title="Normal Distribution", xlabel="x", ylabel="Height")
plt.show()

display(Markdown("###### Normal Distribution (Line)"))
rng = np.random.RandomState(42)
x = rng.randn(1000000)
size = 1000
bins, b = pd.cut(x, bins=size, retbins=True)
avg = [np.mean([b[i], b[i+1]]) for i in range(size)]
fig, ax = plt.subplots(figsize=(20,5))
ax.plot(avg, bins.value_counts())
ax.set(title="Normal Distribution", xlabel="x", ylabel="Height")
plt.show()

display(Markdown("###### 25% Pie Chart"))
data = np.full(4, .25)
fig, ax = plt.subplots(figsize=(5,5))
ax.pie(data, explode=[0,0,0,.25], autopct="%1.2f%%", startangle=45, shadow=True)
ax.set(title="25% Pie Chart")
plt.show()

display(Markdown("###### Random Triangles"))
rng = np.random.RandomState(42)
points = 40
x = rng.randn(points)
y = rng.randn(points)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x, y, s=(x**2)*200, c=y**2, cmap="rainbow", marker="^", alpha=.8)
plt.show()