import numpy as np

# 10 0's
print(np.zeros(10))
print()

# 4 2's
print(np.full(4, 2))
print()

# 3 by 4 range
print(np.arange(1, 13).reshape((3,4)))
print()

# Linspace
print(np.linspace(2, 438, 17))
print()

# Squares
print(np.arange(1, 26) ** 2)
print()

# Random values below .3
rng = np.random.RandomState(453)
values = rng.rand(12)
print(values[values < .3])
print()

# Mean of random values
print(np.random.RandomState(2351).rand(10000000).mean())
print()

# Random values above .999
rng = np.random.RandomState(57963)
values = rng.rand(100000000)
print(f"value > .999 ? {np.any(values > .999)}")
print(f"# values > .999 = {np.sum(values > .999)}")