# Numbers 0 to 10
print([x for x in range(0,11)])
print()

# Even numbers 0 to 10
print([x for x in range(0,11) if x % 2 == 0])
print()

# Odd numbers 0 to 10, not the implicit cast of 1 == True is used
print([x for x in range(1,11) if x % 2])
print()

# Squares 0 to 10
print([x * x for x in range(0,11)])
print()

# Square roots 0 to 10
import math # this line is not necessary, if math is already imported
print([math.sqrt(x) for x in range(0, 11)])
print()

# Numbers 0 to 100 that are divisable by 3 and 5
print([x for x in range(0, 101) if x % 3 == 0 and x % 5 == 0])
print()

# 5 random integers between 0 and 1000
import random as rnd
print([rnd.randint(0,1001) for _ in range(5)])