# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown
# Init Solution completed
# display() & Markdown() are just needed so the solutions can be visualized. They are not expected in your code.


display(Markdown("##### Loading Data"))
airbnbData = pd.read_csv("./Ex04_04_Data.csv")
display(airbnbData.head(3))


display(Markdown("##### Binning to Quartiles"))
display(pd.qcut(airbnbData["number_of_reviews"], q=4))


display(Markdown("##### 5000 Samples"))
display(airbnbData.sample(5000))


display(Markdown("##### 25 Samples per Neighborhood"))
display(airbnbData.groupby("neighbourhood_group").apply(lambda g: g.sample(25)))