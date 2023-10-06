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
data = pd.read_csv("./Ex13_DataViz_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Listing Regions"))
regions = pd.DataFrame(data["Region"].unique(), columns=["Region"])
display(regions)
