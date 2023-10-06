# Your code does not need to contain the display()-wrapper
from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()


fig, ax = plt.subplots()
ax.plot([0,10], [2,2])
ax.set(title="Horizontal Line @ y=2")

fig, ax = plt.subplots()
ax.plot([5,5], [-1, 7.5], ":r")
ax.set(title="Vertical Line @ x=5")

import math
x = np.linspace(-math.pi, 5*math.pi, 100)
fig, ax = plt.subplots()
ax.plot(x, np.cos(x), color="g")
ax.set(title="cos(x)")

rng = np.random.RandomState(42)
x = rng.rand(50)*500 - 250
y = rng.rand(50)*500 - 250
fig, ax = plt.subplots()
ax.plot(x, y, "-.k")
ax.set(title="Random Lines")

x = np.linspace(0,20, 100)
fig, ax = plt.subplots()
ax.plot(x, x**3)
ax.set(ylim=[100, 7000])
ax.set(title="y=x^3")