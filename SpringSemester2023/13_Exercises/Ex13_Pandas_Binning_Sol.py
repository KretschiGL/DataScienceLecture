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
data = pd.read_csv("./Ex13_Pandas_Binning_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### 4 Equi-Width Bins"))
equi4 = pd.cut(data["Education"], bins=4).value_counts()
display(equi4)

display(Markdown("##### 5 Equi-Width Bins"))
equi5 = pd.cut(data["Age"], bins=[0,20,40,60,80,100]).value_counts(sort=False)
display(equi5)

display(Markdown("##### 6 Equi-Depth Bins"))
equi6 = pd.qcut(data["Age"], q=6).value_counts()
display(equi6)

display(Markdown("##### Income Inequality"))
equi_in = pd.qcut(data["Income"], q=[0,.1,.33,.5,.75,.9,.99,1]).value_counts(sort=False)
display(equi_in)
