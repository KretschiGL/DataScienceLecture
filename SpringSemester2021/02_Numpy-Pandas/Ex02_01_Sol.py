# Using the display function allows to show data frames as html even it's not the last entry (using print looks ugly)
# Your code does not need to contain the display()-wrapper
from IPython.display import display

# Load
data = pd.read_csv("./Ex02_01_Data.csv")

# Show structure
display(data.info())

# Show basic stats
display(data.describe())

# Show top 10
display(data.head(10))

# Show last 3
display(data.tail(3))

# Get Switzerland
swiss = data[data["Country or region"] == "Switzerland"]
display(swiss)

# Get gap to top nation
display(data["Score"].max() - swiss["Score"])

# Get difference to unhappiest nation
display(swiss["Score"] - data["Score"].min())

# Nation with highest GDP contribution
display(data[data["GDP per capita"] == data["GDP per capita"].max()])

# Normalized score
data["IntScore"] = np.int32(data["Score"])
display(data.head(5))

# Grouping
scoreGroups = data.groupby("IntScore")

# Countries per group
display(scoreGroups["Country or region"].count())

# Min, mean, median and max per group
display(scoreGroups["GDP per capita"].aggregate([np.min, np.mean, np.median, np.max]))

# Showing top and bottom of each group
top = scoreGroups["Overall rank"].min()
bottom = scoreGroups["Overall rank"].max()
display(data[data["Overall rank"].isin(top.values) | data["Overall rank"].isin(bottom.values)])