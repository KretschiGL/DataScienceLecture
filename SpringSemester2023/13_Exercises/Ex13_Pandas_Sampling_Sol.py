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
data = pd.read_csv("./Ex13_Pandas_Sampling_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Ploting Ethnicities"))
data["Ethnicity"].value_counts().plot.bar(rot=0)
plt.show()

display(Markdown("##### Sampling Caucasian"))
data_cau = data[data["Ethnicity"] == "Caucasian"].sample(150)
display(data_cau.head(5))

display(Markdown("##### Sampling Asian"))
data_asi = data[data["Ethnicity"] == "Asian"].sample(150, replace=True)
display(data_asi.head(5))

display(Markdown("##### Sampling African American"))
data_afr = data[data["Ethnicity"] == "African American"].sample(150, replace=True)
display(data_afr.head(5))

display(Markdown("##### Fixed Dataset"))
data_fix = pd.concat([data_cau, data_asi, data_afr])
data_fix["Ethnicity"].value_counts().plot.bar(rot=0)
plt.show()
