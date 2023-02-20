# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### Preprocessing a Data Set for an MBA"))
data = pd.read_csv("./Ex11_MBA_Data_Raw.csv", sep=";", header=None)
display(Markdown("###### Raw Data"))
display(data.head(5))

data = data.stack()
data = data.reset_index().rename(columns={"level_0":"Cart", 0:"Item"}).drop("level_1", axis=1)
data["Item"] = data["Item"].str.strip()
data["Amount"] = 1
data = data.groupby(["Cart", "Item"])["Amount"].sum().unstack().fillna(0).astype(bool).astype(int)
display(Markdown("###### Processed Data"))
display(data.head(10))
