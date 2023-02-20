# Sum of list
def sum(values):
    result = 0
    for v in values:
        result += v
    return result

values = list(range(0,11))
s = sum(values)
print(s)
print()

# Square list to dictionary
def sqrDict(values):
    result = {}
    for v in values:
        result[v] = v * v
    return result

values = list(range(0,11))
squares = sqrDict(values)
print(squares)
print()

# Apply lambda on list
def map(l, func):
    result = []
    for item in l:
        result.append(func(item))
    return result

items = list(range(0,10,2))
oneHigher = map(items, lambda i : i + 1)
print(oneHigher)
print()

