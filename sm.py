import statsmodels.api as sm
import numpy as np
duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = [3, 5, 7, 9]
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

print(results.summary())