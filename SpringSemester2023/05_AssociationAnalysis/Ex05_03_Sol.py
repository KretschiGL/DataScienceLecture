# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### Loading"))
data = pd.read_csv("./Ex05_03_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Processing"))
display(Markdown("###### Step 6: Grouping"))
stackedData = data.groupby(["Country", "InvoiceNo", "Description"])["Quantity"].sum()
display(stackedData.head(5))
display(Markdown("###### Step 7: Unstacking"))
unstackedData = stackedData.unstack()
display(unstackedData.head(5))
display(Markdown("###### Step 8: Normalizing"))
fixedData = unstackedData.fillna(0).astype(bool).astype(int)
display(fixedData.head(5))

# Storing to *.csv
display(Markdown("Storing to file..."))
fixedData = fixedData.replace(0, np.nan)
fixedData.to_csv("./Ex05_03_Output.csv", sep=";")
display(Markdown("Storing completed"))