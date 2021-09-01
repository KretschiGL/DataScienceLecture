# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### 1000 Randoms"))
rng = np.random.RandomState(42)
n = rng.normal(5,1,size=1000)
display(n[:5])

display(Markdown("##### Min/Max"))
display(f"Max: {np.max(n)}")
display(f"Min: {np.min(n)}")

display(Markdown("##### Mean/StdDev/Median"))
display(f"Mean: {np.mean(n)}")
display(f"StdDev: {np.std(n)}")
display(f"Median: {np.median(n)}")

display(Markdown("##### Quartiles"))
display(f"25%: {np.quantile(n, .25)}")
display(f"50%: {np.quantile(n, .5)}")
display(f"75%: {np.quantile(n, .75)}")

display(Markdown("##### Top"))
display(f"90%: {np.quantile(n, .9)}")
display(f"99%: {np.quantile(n, .99)}")
