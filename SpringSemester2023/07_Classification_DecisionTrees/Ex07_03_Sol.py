# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex07_03_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Checking for NaN values"))
display(data.isna().sum())

display(Markdown("##### Fixing NaN values in *children* column"))
data["children"] = data["children"].fillna(0).astype(int)
display(Markdown(f"Total NaN: {data.isna().sum().sum()}"))

display(Markdown("##### Info"))
display(data.info())

display(Markdown("##### Replacing"))
display(Markdown("- *hotel*"))
data = pd.concat([data.drop("hotel", axis=1), pd.get_dummies(data["hotel"])], axis="columns")
display(data.head(5))

display(Markdown("- *arriaval_date_month*"))
import calendar
months = dict((v,k) for k,v in enumerate(calendar.month_name))
del(months[""])
data["arrival_date_month"] = data["arrival_date_month"].replace(months).astype(int)
display(data.head(5))

display(Markdown("- *meal*"))
data = pd.concat([data.drop("meal", axis="columns"), pd.get_dummies(data["meal"])], axis="columns")
display(data.head(5))

display(Markdown("- *distribution_channel*"))
data = pd.concat([data.drop("distribution_channel", axis="columns"), pd.get_dummies(data["distribution_channel"], prefix="dist_ch")], axis="columns")
display(data.head(5))

display(Markdown("- *deposit_type*"))
data = pd.concat([data.drop("deposit_type", axis="columns"), pd.get_dummies(data["deposit_type"])], axis="columns")
display(data.head(5))

display(Markdown("- *customer_type*"))
data = pd.concat([data.drop("customer_type", axis="columns"), pd.get_dummies(data["customer_type"])], axis="columns")
display(data.head(5))

display(Markdown("##### Info"))
display(data.info())

display(Markdown("##### Replacing Rooms"))
import string
rooms = dict(zip(string.ascii_uppercase, range(1,27)))
rooms
data["reserved_room_type"] = data["reserved_room_type"].replace(rooms)
data["assigned_room_type"] = data["assigned_room_type"].replace(rooms)
display(data.head(5)[["reserved_room_type", "assigned_room_type"]])

display(Markdown("##### Final Check"))
display(data.info())