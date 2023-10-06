# Your code does not need to contain the display()-wrapper
from IPython.display import display, Markdown

import numpy as np
import pandas as pd

display(Markdown("##### Loading data"))
original = pd.read_csv("./Ex03_02_Data.csv")
games = original.copy()
display(games.head(5))

display(Markdown("##### Checking how many values are missing"))
display(games.isna().mean())

display(Markdown("##### Replacing Developers with Publishers"))
games["Developer"] = games["Developer"].fillna(games["Publisher"])
display(games.isna().mean())

display(Markdown("##### Replacing Critic Count"))
games["Critic_Count"] = games["Critic_Count"].fillna(games["Critic_Count"].mean())
display(games.isna().mean())

display(Markdown("##### Replacing the User Score"))
games["User_Score"] = games["User_Score"].fillna(games["Global_Sales"]/100 + games["User_Score"].median())
display(games.isna().mean())

display(Markdown("##### Replacing Release Years"))
releaseDates = games.groupby("Platform", group_keys=False)["Year_of_Release"]
games["Year_of_Release"] = releaseDates.apply(lambda g : g.fillna(g.median()))
display(games.isna().mean())

display(Markdown("##### Replacing the Critic Score"))
display(Markdown("Platform, Genre & Publisher"))
scores = games.groupby(["Platform", "Genre", "Publisher"], group_keys=False)["Critic_Score"]
games["Critic_Score"] = scores.apply(lambda g : g.fillna(g.median()))
display(games.isna().mean())

display(Markdown("Genre & Publisher"))
scores = games.groupby(["Genre", "Publisher"], group_keys=False)["Critic_Score"]
games["Critic_Score"] = scores.apply(lambda g : g.fillna(g.median()))
display(games.isna().mean())

display(Markdown("Genre"))
scores = games.groupby("Genre", group_keys=False)["Critic_Score"]
games["Critic_Score"] = scores.apply(lambda g : g.fillna(g.median()))
display(games.isna().mean())

display(Markdown("##### Check if the data set has no missing values"))
display(games.isna().any())