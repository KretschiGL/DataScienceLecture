# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
# Init Solution completed



labels = ["1/3", "1/3", "1/3"]
sizes = [1/3, 1/3, 1/3]
explode = [.1, .1, .1]

fig, ax = plt.subplots()
fig.suptitle("Fair Share")
ax.pie(sizes, explode=explode, labels=labels, autopct="%1.3f%%", shadow=True)



labels = ["1/2", "1/4", "1/8", "1/8"]
sizes = [.5, .25, .125, .125]
explode = [.1, 0, 0, 0]

fig, ax = plt.subplots()
fig.suptitle("50/50")
ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)