# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### 3 Random Numbers"))
rng = np.random.RandomState(42)
n = rng.rand(3)
display(n)

display(Markdown("##### 2 RNG's"))
rng1 = np.random.RandomState(42)
display(f"RNG1: {rng1.normal(size=5)}")
rng2 = np.random.RandomState(42)
display(f"RNG2: {rng2.normal(size=5)}")

display(Markdown("##### 10 Integers"))
rng = np.random.RandomState(42)
n = rng.uniform(-10,10,size=10).round()
display(n)

display(Markdown("##### 4x4 Randoms"))
rng = np.random.RandomState(42)
grid = rng.normal(0,2,size=(4,4))
display(grid)
