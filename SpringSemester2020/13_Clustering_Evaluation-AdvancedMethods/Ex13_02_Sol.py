# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex13_02_Data.csv")
display(data.head(5))

display(Markdown("##### Pairplot"))
sns.pairplot(data)
plt.show()

display(Markdown("##### Pairplot (color = Cylinders)"))
sns.pairplot(data, hue="cylinders")
plt.show()

display(Markdown("##### Horsepower vs MPG"))
fig, ax = plt.subplots(figsize=(10,10))
sns.kdeplot(data["horsepower"], data["mpg"], ax=ax)
plt.show()

display(Markdown("##### Acceleration vs Weight"))
fig, ax = plt.subplots(figsize=(10,10))
data.plot.scatter("acceleration", "weight", ax=ax, c="cylinders", cmap="rainbow", alpha=.5)
sns.kdeplot(data["acceleration"], data["weight"], ax=ax)
plt.show()

display(Markdown("##### Weight vs Acceleration"))
sns.jointplot(data=data, x="weight", y="acceleration")
plt.show()

display(Markdown("##### Weight vs Acceleration (KDE)"))
sns.jointplot(data=data, x="weight", y="acceleration", kind="kde")
plt.show()

display(Markdown("##### Displacement vs Horsepower"))
sns.jointplot(data=data, x="displacement", y="horsepower", kind="hex")
plt.show()