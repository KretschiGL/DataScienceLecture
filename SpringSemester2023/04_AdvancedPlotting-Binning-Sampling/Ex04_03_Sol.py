# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display
# Init Solution completed


rng = np.random.RandomState(42)
data = rng.uniform(-100, 100, 10000)
fig, ax = plt.subplots(figsize=(20,5))
ax.set(title="Uniform Distribution Histogram")
ax.hist(data, bins=101)


rng = np.random.RandomState(42)
data = rng.normal(5,3, 100000)
display(print(np.mean(data)))
fig, ax = plt.subplots()
ax.set(title="Normal(5,3) Histogram")
ax.hist(data, bins=80)