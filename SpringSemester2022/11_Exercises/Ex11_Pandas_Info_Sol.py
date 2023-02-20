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
data = pd.read_csv("./Ex11_Pandas_Info_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Showing Info"))
display(data.info())

display(Markdown("##### Showing the Summary"))
display(data.describe(include="all"))
