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
data = pd.read_csv("./Ex10_Pandas_Missing.csv")
display(data.head(5))

display(Markdown("##### Showing the Missing Values"))
missing = pd.merge(pd.Series(data.isna().mean(), name="% Missing"), pd.Series(data.isna().sum(), name="# Missing"), left_index=True, right_index=True)
display(missing.head(10))

display(Markdown("##### Replacing Accelerations"))
data["acceleration"] = data["acceleration"].fillna(16)
missing_acc = data["acceleration"].isna().sum()
display(Markdown(f"Missing Accelerations: {missing_acc}"))

display(Markdown("##### Replacing Years"))
data["year"] = data["year"].fillna(method="ffill").fillna(method="bfill")
missing_year = data["year"].isna().sum()
display(Markdown(f"Missing Years: {missing_year}"))

display(Markdown("##### Replacing Weights"))
data["weight"] = data.groupby(["mpg"])["weight"].apply(lambda g : g.fillna(g.mean()))
missing_weight =data["weight"].isna().sum()
display(Markdown(f"Remaining Missing Weights: {missing_weight}"))

data["weight"] = data["weight"].fillna(data["weight"].median())
missing_weight = data["weight"].isna().sum()
display(Markdown(f"Missing Weights: {missing_weight}"))

display(Markdown("##### Replacing Displacements"))
data["displacement"] = data.groupby(["cylinders", "horsepower"])["displacement"].apply(lambda d : d.fillna(d.mean()))
data["displacement"] = data.groupby(["cylinders"])["displacement"].apply(lambda d : d.fillna(d.mean()))
missing_displacement = data["displacement"].isna().sum()
display(Markdown(f"Missing Displacements: {missing_displacement}"))

display(Markdown("##### Final Check"))
check = data.isna().mean()
display(check)
