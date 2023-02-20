# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### 0-9"))
n = np.arange(0,10)
display(n)

display(Markdown("##### n < 5"))
b = n < 5
display(b)

display(Markdown("##### All < 10"))
b = np.all(n <= 9)
display(b)

display(Markdown("##### Any > 9"))
b = np.any(n > 9)
display(b)
