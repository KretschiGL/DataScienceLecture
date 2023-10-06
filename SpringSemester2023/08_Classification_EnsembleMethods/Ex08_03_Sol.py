# Init Solution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

from IPython.display import display, Markdown
# Init Solution completed

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

display(Markdown("##### Loading Data"))
data = pd.read_csv("./Ex08_03_Data.csv", sep=";")
display(data.head(5))

display(Markdown("##### Class Imbalance"))
data["Revenue"].value_counts().plot.bar()
plt.show()

display(Markdown("##### Fixing Class Imbalance"))
revenue_true = data[data["Revenue"] == True].sample(10000, replace=True, random_state=42)
revenue_false = data[data["Revenue"] == False]
data_rev = pd.concat([revenue_true, revenue_false])
data_rev["Revenue"].value_counts().plot.bar()
plt.show()

display(Markdown("##### Datasets"))
features = data_rev.columns.drop("Revenue").values
X = data_rev[features]
y = data_rev["Revenue"]
display(X.head(5))
display(y.head(5))

display(Markdown("##### Grid Search"))
display(Markdown("This takes a while..."))
params = {
    "max_features":[8, 16, "sqrt", None],
    "max_samples":[.8, None],
    "min_samples_split":[2, 10, 18]
}
grid = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42), params, cv=5, n_jobs=4)
grid.fit(X, y)
display(grid)

display(Markdown("##### Hyperparameter values"))
display(grid.best_params_)

display(Markdown("##### Best Model Score"))
display(grid.best_score_)

display(Markdown("##### Creating & Training Model"))
model = grid.best_estimator_
model.fit(X, y)
display(model)

display(Markdown("##### Loading new Data"))
data_use = pd.read_csv("./Ex08_03_Data_Use.csv", sep=";")
display(data_use.head(5))

display(Markdown("##### Predicting Probability of Purchase"))
proba = model.predict_proba(data_use)
proba = pd.DataFrame(proba, columns=["No Purchase", "Purchase_Probability"])
data_pur = pd.concat([proba["Purchase_Probability"], data_use], axis="columns")
display(data_pur.head(10))