# Add 5 to number
add5 = lambda n : n + 5
print(add5(2))
print(add5(7))
print()

# Square number
sqr = lambda n : n * n
print(sqr(2))
print(sqr(7))
print()

# Next integer
nextInt = lambda n : int(n) + 1
print(nextInt(2.7))
print(nextInt(7.2))
print()

# Previous integer of half
prevInt = lambda n : int(n // 2)
print(prevInt(2.7))
print(prevInt(7.2))
print()

# Division lambda
div = lambda dvsr : lambda dvdn : dvdn / dvsr
print(div(5)(10))
print(div(3)(27))