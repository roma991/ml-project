import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
print(pd.read_csv('house_prices.csv').info())