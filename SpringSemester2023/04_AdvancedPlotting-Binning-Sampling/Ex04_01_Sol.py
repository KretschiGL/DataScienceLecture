# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
# Init Solution completed


fig, ax = plt.subplots()
ax.set(title="Weather Summary")
ax.bar(category, days)


idx = np.argsort(days)
fig, ax = plt.subplots()
ax.set(title="Ordered Weather Conditions by Occurance")
ax.bar(category[idx], days[idx])


idx = np.argsort(category)[::-1]
fig, ax = plt.subplots()
ax.set(title="Weather Conditions by Name")
ax.barh(category[idx], days[idx])