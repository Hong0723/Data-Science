import pandas as pd
import numpy as np

sales = pd.Series([781, 650, 705, 406, 580, 450, 550, 640])
sales
sales.index
sales.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
sales

sales.size

sales.iloc[1]
sales.loc['F']
sales.iloc[3:6]
sales.loc[['A', 'B', "D"]]
sales.loc[(sales < 500) | (sales > 700)] 
LargeB = sales.loc['B']
sales.where(sales > LargeB ).dropna()
sales.where(sales < 600).dropna()
sales.loc[sales < 600] * 1.2
sales.mean()
sales.sum()
sales.std()

sales.loc[['A','C']] = [810,820]
sales
sales.loc['J'] = 400
sales
sales = sales.drop('J')
sales

sales2 = sales
sales2 = sales2[ : ]+ 500
sales
sales2


