# Fetch data set from: https://archive.ics.uci.edu/dataset/20/census+income
# and save it to the file system

import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
census_income = fetch_ucirepo(id=20)

# data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)

# Combine features + target in same DataFrame
df = pd.concat([X, y], axis=1)

# Save to file
df.to_csv("data/census_income.csv", index=False)
