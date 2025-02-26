import pandas as pd

df= pd.read_csv('airquality.csv')
df

df.isnull().sum() # 컬럼별 결측값 확인

df = df.dropna()
df2 = df.reset_index(drop = True)
df2.head()

df = df.reset_index(drop =True) 
df.head()

import pandas as pd
from sklearn.impute import KNNImputer 

tmp_np = df.iloc[:,:5].to_numpy() 
imputer = KNNImputer(n_neighbors=6)
tmp_np = imputer.fit_transform(tmp_np)
df.iloc[:,:5] = tmp_np
df3 = df.copy()
df3.head()

import numpy as np
from scipy import stats

df = pd.read_csv('airquality.csv')

wind = df.Wind
z = np.abs(stats.zscore(wind))
outliers = wind[z > 2]
print(outliers)

df_sorted = df.sort_values('Temp', ascending = False)
df4 = df_sorted 
df4.head()

df5 = df.sample(n = 10, random_state= 123)
df5


df2 

df_agg = df2.groupby(['Month'])[['Wind','Temp']].max()
df_agg

import pandas as pd
df= pd.read_csv('mtcars.csv')
p1 = df.pivot_table(index= 'carb',
                    columns = 'gear',
                    values = 'mpg',
                    aggfunc = 'mean')
p1.head()