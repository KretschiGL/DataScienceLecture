# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex12_01_Data.csv")
display(data.head(5))

display(Markdown("##### New Dataset"))
data_new = pd.DataFrame(data[["name", "year"]])
display(data_new.head(5))

display(Markdown("##### mpg -> MinMaxScaler(-1,1)"))
minmax = MinMaxScaler(feature_range=(-1,1))
data_new["mpg"] = minmax.fit_transform(data["mpg"].to_numpy().reshape(-1,1))
display(data_new.head(5))

display(Markdown("##### cylinders -> MinMaxScaler(-2,2)"))
minmax = MinMaxScaler(feature_range=(-2,2))
data_new["cylinders"] = minmax.fit_transform(data["cylinders"].to_numpy().reshape(-1,1))
display(data_new.head(5))

display(Markdown("##### horsepower -> StandardScaler"))
std = StandardScaler()
data_new["horsepower"] = std.fit_transform(data["horsepower"].to_numpy().reshape(-1,1))
display(data_new.head(5))

display(Markdown("##### acceleration -> StandardScaler"))
std = StandardScaler()
data_new["acceleration"] = std.fit_transform(data["acceleration"].to_numpy().reshape(-1,1))
display(data_new.head(5))

display(Markdown("##### displacement & weight -> RobustScaler"))
robust = RobustScaler()
data_new["displacement"] = robust.fit_transform(data["displacement"].to_numpy().reshape(-1,1))
data_new["weight"] = robust.fit_transform(data["weight"].to_numpy().reshape(-1,1))
display(data_new.head(5))
