# Your code does not need to contain the display()-wrapper
from IPython.display import display, Markdown

import pandas as pd

display(Markdown("##### Loading"))
games = pd.read_csv("./Ex03_01_Data.csv")
display(games.head(5))

display(Markdown("##### Percentage of missing values"))
display(games.isna().mean())

display(Markdown("##### Every complete row"))
minimized = games.dropna()
display(len(minimized))

display(Markdown("##### All columns without missing values"))
display(games.dropna(axis=1))

display(Markdown("##### All games with names"))
display(games[games["Name"].notna()])

display(Markdown("##### Drop games with missing critic score"))
noCritic = games[games["Critic_Score"].notna()]
display(noCritic["Critic_Score"].isna().sum())

display(Markdown("##### Drop Developer column"))
display(games.drop("Developer", axis=1))

display(Markdown("##### Games with user score of 8 and higher"))
display(games[games["User_Score"] >= 8])

display(Markdown("##### Games sold in europe"))
display(games[games["EU_Sales"] > 0])

display(Markdown("##### Columns where less than 40% of values are missing"))
display(games[games.columns[games.isna().mean() < .4]])